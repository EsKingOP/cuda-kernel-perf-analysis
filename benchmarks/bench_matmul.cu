/**
 * bench_matmul.cu
 *
 * Full benchmark harness for all matmul implementations across a range of
 * matrix sizes.
 *
 * Implementations benchmarked
 * ───────────────────────────
 *   cpu_naive          — triple-loop ijk (single-threaded)
 *   cpu_cache_friendly — loop-reordered ikj (single-threaded)
 *   cuda_naive         — naive global-memory CUDA kernel (block 16×16)
 *   cuda_tiled_8       — shared-memory tiled CUDA (tile 8×8)
 *   cuda_tiled_16      — shared-memory tiled CUDA (tile 16×16)
 *   cuda_tiled_32      — shared-memory tiled CUDA (tile 32×32)
 *
 * Matrix sizes: 256, 512, 1024, 2048, 4096 (square, single-precision float)
 * Runs per config: NUM_RUNS (default 10) — first run is a warm-up
 *
 * Output
 * ──────
 *   Console: human-readable stats table
 *   File   : results/bench_matmul.csv  (all raw per-run rows)
 *
 * Correctness
 * ───────────
 *   All GPU results are compared element-wise against the CPU cache-friendly
 *   reference.  A discrepancy > 1e-3 is flagged as FAIL.
 *
 * Timing
 * ──────
 *   GPU kernel time is measured with cudaEvent_t (device-side timer, avoids
 *   CPU-GPU synchronisation overhead that inflates std::chrono measurements).
 *   Transfer time (H2D + D2H) is measured separately.
 *   CPU time is measured with std::chrono::high_resolution_clock.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#include "matrix_utils.h"
#include "perf_metrics.h"
#include "../src/cpu/matmul_cpu.h"
#include "../src/cuda/matmul_tiled.cuh"

// ── Forward declaration from matmul_naive.cu ─────────────────────────────────
void matmul_naive_cuda(const float* h_A, const float* h_B, float* h_C,
                       int M, int K, int N,
                       int block_dim,
                       double* kernel_ms, double* transfer_ms);

// ── Configuration ─────────────────────────────────────────────────────────────
static const int SIZES[]      = {256, 512, 1024, 2048, 4096};
static const int NUM_SIZES    = static_cast<int>(sizeof(SIZES) / sizeof(SIZES[0]));
static const int NUM_RUNS     = 10;
static const int NAIVE_BLOCK  = 16;
static const int TILE_SIZES[] = {8, 16, 32};

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    // Print device info
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (SM %d.%d, %d SMs, %.1f GB VRAM)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);
    printf("Peak DRAM bandwidth: %.1f GB/s\n\n",
           (prop.memoryBusWidth / 8.0) * prop.memoryClockRate * 2.0 * 1e-6);

    // Open CSV output
    FILE* csv = fopen("results/bench_matmul.csv", "w");
    if (!csv) { perror("fopen results/bench_matmul.csv"); return 1; }
    csv_write_header(csv);

    std::vector<PerfStats> all_stats;

    for (int si = 0; si < NUM_SIZES; ++si) {
        const int N = SIZES[si];
        const double flops = matmul_flops(N, N, N);
        // Approx bytes transferred for BW calculation (A + B reads + C write)
        const double bytes_traffic =
            3.0 * sizeof(float) * static_cast<double>(N) * N;

        printf("────── N = %d ──────\n", N);

        // ── Allocate host matrices ────────────────────────────────────────
        Matrix A = matrix_alloc_pinned(N, N);
        Matrix B = matrix_alloc_pinned(N, N);
        Matrix C_ref  = matrix_alloc(N, N);   // CPU reference
        Matrix C_gpu  = matrix_alloc_pinned(N, N);

        matrix_fill_random(A, -1.0f, 1.0f, 1234);
        matrix_fill_random(B, -1.0f, 1.0f, 5678);

        // ── CPU baseline (skip for large N due to time) ───────────────────
        // Always run cpu_cache_friendly for correctness reference.
        {
            std::vector<PerfResult> runs;
            bool skip_naive_cpu = (N > 1024);

            // cpu_cache_friendly (always run — used as correctness ref)
            for (int r = 0; r < NUM_RUNS; ++r) {
                std::memset(C_ref.data, 0, C_ref.bytes());
                WallTimer timer;
                timer.start();
                matmul_cpu_cache_friendly(A.data, B.data, C_ref.data, N, N, N);
                double ms = timer.elapsed_ms();

                PerfResult res{};
                res.impl_name   = "cpu_cache_friendly";
                res.matrix_size = N;
                res.run_index   = r;
                res.kernel_ms   = ms;
                res.transfer_ms = 0.0;
                res.total_ms    = ms;
                res.gflops      = gflops(flops, ms);
                res.bandwidth_gb_s = 0.0;
                res.correct     = true;
                res.tile_size   = 0;
                runs.push_back(res);
                csv_write_row(csv, res);
            }
            all_stats.push_back(compute_stats(runs));

            // cpu_naive (skip for N > 1024 — too slow)
            if (!skip_naive_cpu) {
                runs.clear();
                Matrix C_naive_cpu = matrix_alloc(N, N);
                for (int r = 0; r < NUM_RUNS; ++r) {
                    std::memset(C_naive_cpu.data, 0, C_naive_cpu.bytes());
                    WallTimer timer;
                    timer.start();
                    matmul_cpu_naive(A.data, B.data, C_naive_cpu.data, N, N, N);
                    double ms = timer.elapsed_ms();

                    PerfResult res{};
                    res.impl_name   = "cpu_naive";
                    res.matrix_size = N;
                    res.run_index   = r;
                    res.kernel_ms   = ms;
                    res.transfer_ms = 0.0;
                    res.total_ms    = ms;
                    res.gflops      = gflops(flops, ms);
                    res.bandwidth_gb_s = 0.0;
                    res.correct     = (r == 0)
                        ? matrix_verify(C_ref, C_naive_cpu)
                        : runs[0].correct;
                    res.tile_size   = 0;
                    runs.push_back(res);
                    csv_write_row(csv, res);
                }
                all_stats.push_back(compute_stats(runs));
                matrix_free(C_naive_cpu);
            }
        }

        // ── CUDA naive ────────────────────────────────────────────────────
        {
            std::vector<PerfResult> runs;
            for (int r = 0; r < NUM_RUNS; ++r) {
                double k_ms = 0.0, t_ms = 0.0;
                matmul_naive_cuda(A.data, B.data, C_gpu.data,
                                  N, N, N, NAIVE_BLOCK, &k_ms, &t_ms);

                PerfResult res{};
                res.impl_name      = "cuda_naive";
                res.matrix_size    = N;
                res.run_index      = r;
                res.kernel_ms      = k_ms;
                res.transfer_ms    = t_ms;
                res.total_ms       = k_ms + t_ms;
                res.gflops         = gflops(flops, k_ms);
                res.bandwidth_gb_s = (k_ms > 0)
                    ? (bytes_traffic / k_ms) * 1e-6 : 0.0;
                res.correct        = (r == 0)
                    ? matrix_verify(C_ref, C_gpu)
                    : runs[0].correct;
                res.tile_size      = 0;
                runs.push_back(res);
                csv_write_row(csv, res);
            }
            all_stats.push_back(compute_stats(runs));
        }

        // ── CUDA tiled (tile sizes 8, 16, 32) ────────────────────────────
        for (int tile : TILE_SIZES) {
            std::vector<PerfResult> runs;
            char impl_name[32];
            snprintf(impl_name, sizeof(impl_name), "cuda_tiled_%d", tile);

            for (int r = 0; r < NUM_RUNS; ++r) {
                double k_ms = 0.0, t_ms = 0.0;
                matmul_tiled_cuda(A.data, B.data, C_gpu.data,
                                  N, N, N, tile, &k_ms, &t_ms);

                PerfResult res{};
                res.impl_name      = impl_name;
                res.matrix_size    = N;
                res.run_index      = r;
                res.kernel_ms      = k_ms;
                res.transfer_ms    = t_ms;
                res.total_ms       = k_ms + t_ms;
                res.gflops         = gflops(flops, k_ms);
                res.bandwidth_gb_s = (k_ms > 0)
                    ? (bytes_traffic / k_ms) * 1e-6 : 0.0;
                res.correct        = (r == 0)
                    ? matrix_verify(C_ref, C_gpu)
                    : runs[0].correct;
                res.tile_size      = tile;
                runs.push_back(res);
                csv_write_row(csv, res);
            }
            all_stats.push_back(compute_stats(runs));
        }

        // ── Cleanup ───────────────────────────────────────────────────────
        matrix_free_pinned(A);
        matrix_free_pinned(B);
        matrix_free(C_ref);
        matrix_free_pinned(C_gpu);
    }

    fclose(csv);

    print_stats_table(all_stats);
    printf("Results written to results/bench_matmul.csv\n");
    return 0;
}
