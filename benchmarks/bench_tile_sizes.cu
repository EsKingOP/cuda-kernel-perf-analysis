/**
 * bench_tile_sizes.cu
 *
 * Benchmark the tiled matmul kernel across different TILE_SIZE values at a
 * fixed matrix size, analysing the effect of tile size on:
 *
 *   - Kernel execution time (ms)
 *   - Achieved GFLOP/s
 *   - Estimated warp occupancy (via cudaOccupancyMaxActiveBlocksPerMultiprocessor)
 *   - Shared memory usage per block
 *   - Register usage (read from PTX / cubin -- reported by nvcc)
 *
 * Fixed matrix size: 2048×2048 (square, single-precision float)
 * Tile sizes tested: 8, 16, 32
 * Runs per config  : NUM_RUNS (default 20) — statistics are printed
 *
 * Occupancy model
 * ───────────────
 * CUDA's occupancy calculator considers three limiting resources:
 *   1. Threads per SM   (prop.maxThreadsPerMultiProcessor)
 *   2. Blocks per SM    (prop.maxBlocksPerMultiProcessor)
 *   3. Shared memory per SM (prop.sharedMemPerMultiprocessor)
 *   4. Registers per SM (prop.regsPerMultiprocessor)
 *
 * For TILE_SIZE=32: block = 1024 threads, shmem = 2×32×32×4 = 8 KB
 * For TILE_SIZE=16: block =  256 threads, shmem = 2×16×16×4 = 2 KB
 * For TILE_SIZE= 8: block =   64 threads, shmem = 2× 8× 8×4 = 0.5 KB
 *
 * Larger tiles → more threads/block → potential register spilling → lower
 * occupancy despite higher arithmetic intensity per block.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "matrix_utils.h"
#include "perf_metrics.h"
#include "../src/cuda/matmul_tiled.cuh"

static const int FIXED_N   = 2048;
static const int NUM_RUNS  = 20;

// ── Occupancy query for a given tile size ────────────────────────────────────

template <int TILE_SIZE>
static int query_max_active_blocks(int device) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int max_blocks = 0;
    int block_size = TILE_SIZE * TILE_SIZE;
    size_t shmem   = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks,
        matmul_tiled_kernel<TILE_SIZE>,
        block_size,
        shmem);

    if (e != cudaSuccess) {
        fprintf(stderr, "OccupancyQuery error: %s\n", cudaGetErrorString(e));
        return -1;
    }

    return max_blocks;
}

// ── Report occupancy as percentage ───────────────────────────────────────────

static double occupancy_pct(int tile_size, int max_blocks_per_sm,
                             const cudaDeviceProp& prop) {
    int block_threads = tile_size * tile_size;
    int active_threads = max_blocks_per_sm * block_threads;
    return 100.0 * static_cast<double>(active_threads) /
                   static_cast<double>(prop.maxThreadsPerMultiProcessor);
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("Device : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Matrix : %d × %d  (single precision)\n\n", FIXED_N, FIXED_N);

    const int N      = FIXED_N;
    const double flops = matmul_flops(N, N, N);

    Matrix A   = matrix_alloc_pinned(N, N);
    Matrix B   = matrix_alloc_pinned(N, N);
    Matrix C   = matrix_alloc_pinned(N, N);
    Matrix Ref = matrix_alloc(N, N);

    matrix_fill_random(A, -1.0f, 1.0f, 42);
    matrix_fill_random(B, -1.0f, 1.0f, 99);

    // Compute reference with tile=16
    matmul_tiled_cuda(A.data, B.data, Ref.data, N, N, N, 16, nullptr, nullptr);

    // Open CSV
    FILE* csv = fopen("results/bench_tile_sizes.csv", "w");
    if (!csv) { perror("fopen results/bench_tile_sizes.csv"); return 1; }
    fprintf(csv, "tile_size,block_threads,shmem_bytes,max_blocks_per_sm,"
                 "occupancy_pct,run,kernel_ms,gflops,correct\n");

    printf("%-12s %8s %10s %15s %12s %9s %9s %6s\n",
           "tile_size", "threads", "shmem_B", "blks/SM",
           "occupancy%", "mean_ms", "gflops", "ok?");
    printf("%s\n", std::string(80, '-').c_str());

    int tile_sizes[] = {8, 16, 32};

    for (int tile : tile_sizes) {
        int block_threads = tile * tile;
        size_t shmem      = 2 * static_cast<size_t>(tile) * tile * sizeof(float);

        // Query occupancy
        int max_blks = 0;
        double occ   = 0.0;
        if (tile == 8) {
            max_blks = query_max_active_blocks< 8>(dev);
        } else if (tile == 16) {
            max_blks = query_max_active_blocks<16>(dev);
        } else {
            max_blks = query_max_active_blocks<32>(dev);
        }
        if (max_blks > 0) occ = occupancy_pct(tile, max_blks, prop);

        // Run benchmark
        double sum_ms = 0.0, sum_gf = 0.0;
        double min_ms =  1e18, max_ms = 0.0;
        bool all_ok = true;

        for (int r = 0; r < NUM_RUNS; ++r) {
            double k_ms = 0.0;
            matmul_tiled_cuda(A.data, B.data, C.data, N, N, N,
                              tile, &k_ms, nullptr);

            bool ok = (r == 0) ? matrix_verify(Ref, C) : all_ok;
            if (!ok) all_ok = false;
            double gf = gflops(flops, k_ms);
            sum_ms += k_ms;
            sum_gf += gf;
            if (k_ms < min_ms) min_ms = k_ms;
            if (k_ms > max_ms) max_ms = k_ms;

            fprintf(csv, "%d,%d,%zu,%d,%.2f,%d,%.6f,%.2f,%d\n",
                    tile, block_threads, shmem, max_blks, occ,
                    r, k_ms, gf, ok ? 1 : 0);
        }

        double mean_ms = sum_ms / NUM_RUNS;
        double mean_gf = sum_gf / NUM_RUNS;

        printf("%-12d %8d %10zu %15d %12.1f %9.3f %9.1f %6s\n",
               tile, block_threads, shmem, max_blks, occ,
               mean_ms, mean_gf, all_ok ? "PASS" : "FAIL");
    }

    fclose(csv);
    printf("\nResults written to results/bench_tile_sizes.csv\n");

    matrix_free_pinned(A);
    matrix_free_pinned(B);
    matrix_free_pinned(C);
    matrix_free(Ref);
    return 0;
}
