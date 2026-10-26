/**
 * matmul_tiled.cu
 *
 * Tiled (shared-memory) CUDA matrix multiplication kernel.
 *
 * Algorithm overview
 * ──────────────────
 * For an N×N matrix multiply C = A * B, the naive approach loads each element
 * of A and B exactly N times from global DRAM.  Tiling amortises this:
 *
 *   1. Divide A, B, and C into square tiles of edge TILE_SIZE.
 *   2. Each thread block is responsible for computing one TILE_SIZE×TILE_SIZE
 *      tile of C.
 *   3. The partial sum is accumulated across K/TILE_SIZE phases:
 *        a. Load the required tile of A into __shared__ memory (sA).
 *        b. Load the required tile of B into __shared__ memory (sB).
 *        c. __syncthreads() — ensure the entire tile is visible to all threads.
 *        d. Compute the partial dot-product using sA and sB (fast, on-chip).
 *        e. __syncthreads() — prevent early overwrite of shared memory.
 *   4. Write the accumulated result to C in global memory.
 *
 * Memory reuse: each element loaded into shared memory is used TILE_SIZE times
 * by threads in the same block, reducing DRAM traffic by a factor of TILE_SIZE.
 *
 * Non-power-of-2 support
 * ──────────────────────
 * When M, K, or N is not divisible by TILE_SIZE, we use a padded load:
 * out-of-bounds threads load 0.0f into shared memory so the arithmetic
 * remains correct, and the write to C is guarded by a bounds check.
 *
 * Template parameter
 * ──────────────────
 * TILE_SIZE is a compile-time constant so the compiler can:
 *   - Unroll the inner accumulation loop.
 *   - Allocate __shared__ arrays with known size (no dynamic shared memory).
 *   - Emit optimal code for each tile size without branching.
 */

#include "matmul_tiled.cuh"
#include "matrix_utils.h"
#include <cstdio>
#include <cstdlib>

// ─── Kernel ──────────────────────────────────────────────────────────────────

template <int TILE_SIZE>
__global__
void matmul_tiled_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                               float* __restrict__ C,
                         int M, int K, int N) {
    // Shared memory tiles — one for each input matrix.
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Global output position for this thread.
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    // Number of tile phases needed to cover the K dimension.
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; ++tile) {
        // ── Load tile of A ────────────────────────────────────────────────
        // Thread (ty, tx) loads A[row, tile*TILE_SIZE + tx]
        const int a_col = tile * TILE_SIZE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // ── Load tile of B ────────────────────────────────────────────────
        // Thread (ty, tx) loads B[tile*TILE_SIZE + ty, col]
        const int b_row = tile * TILE_SIZE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // ── Synchronise: tiles are fully loaded ───────────────────────────
        __syncthreads();

        // ── Accumulate partial dot product ────────────────────────────────
        // Unrolled by the compiler for compile-time TILE_SIZE.
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // ── Synchronise: prevent overwrite before all threads finish ──────
        __syncthreads();
    }

    // ── Write result ──────────────────────────────────────────────────────
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Explicit template instantiations — avoids link errors and controls which
// tile sizes are compiled into the binary.
template __global__
void matmul_tiled_kernel< 8>(const float*, const float*, float*, int, int, int);
template __global__
void matmul_tiled_kernel<16>(const float*, const float*, float*, int, int, int);
template __global__
void matmul_tiled_kernel<32>(const float*, const float*, float*, int, int, int);

// ─── Internal launcher ────────────────────────────────────────────────────────

template <int TILE_SIZE>
static void launch_tiled(const float* d_A, const float* d_B, float* d_C,
                          int M, int K, int N,
                          double* kernel_ms) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    matmul_tiled_kernel<TILE_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "Tiled kernel error (tile=%d): %s\n",
                TILE_SIZE, cudaGetErrorString(kerr));
        std::exit(EXIT_FAILURE);
    }

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    if (kernel_ms) *kernel_ms = static_cast<double>(ms);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
}

// ─── Host wrapper ─────────────────────────────────────────────────────────────

void matmul_tiled_cuda(const float* h_A, const float* h_B, float* h_C,
                       int M, int K, int N,
                       int tile_size,
                       double* kernel_ms, double* transfer_ms) {
    const size_t bytes_A = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t bytes_B = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t bytes_C = sizeof(float) * static_cast<size_t>(M) * N;

    // ── Allocate device memory ────────────────────────────────────────────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes_A));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes_B));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes_C));

    // ── H2D transfer (timed) ──────────────────────────────────────────────
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, t0, t1));

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    // ── Launch kernel (dispatched by tile_size) ───────────────────────────
    double k_ms = 0.0;
    switch (tile_size) {
        case  8: launch_tiled< 8>(d_A, d_B, d_C, M, K, N, &k_ms); break;
        case 16: launch_tiled<16>(d_A, d_B, d_C, M, K, N, &k_ms); break;
        case 32: launch_tiled<32>(d_A, d_B, d_C, M, K, N, &k_ms); break;
        default:
            fprintf(stderr, "Unsupported tile_size=%d (use 8, 16, or 32)\n",
                    tile_size);
            std::exit(EXIT_FAILURE);
    }
    if (kernel_ms) *kernel_ms = k_ms;

    // ── D2H transfer (timed) ──────────────────────────────────────────────
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, t0, t1));

    if (transfer_ms) *transfer_ms = static_cast<double>(h2d_ms + d2h_ms);

    // ── Cleanup ───────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
