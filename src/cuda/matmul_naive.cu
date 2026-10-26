/**
 * matmul_naive.cu
 *
 * Naive CUDA matrix multiplication kernel.
 *
 * Strategy:
 *   - One thread per output element C[row, col].
 *   - All memory accesses go directly to global DRAM — no shared memory.
 *   - Loads from A are coalesced (threads in the same warp read adjacent
 *     columns of A[row, k] across increasing col-indices), but loads from
 *     B[k, col] traverse a column of B, which IS coalesced because threads
 *     in the same warp share the same 'k' and differ in 'col'.
 *   - The main bottleneck is the total DRAM traffic: for an N×N multiply
 *     each output element reads N elements from A and N from B → O(N³)
 *     global loads with no reuse.
 *
 * This file also exposes a host-callable wrapper that handles device memory
 * allocation, kernel launch, and optional timing via cudaEvent_t.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include "matrix_utils.h"

// ─── Kernel ──────────────────────────────────────────────────────────────────

/**
 * Each thread computes one element of C.
 *
 * Grid  : ceil(N/BLOCK_DIM) × ceil(M/BLOCK_DIM)  2-D blocks
 * Block : BLOCK_DIM × BLOCK_DIM threads
 *
 * Bounds check is required when N / M is not a multiple of BLOCK_DIM.
 */
__global__
void matmul_naive_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                               float* __restrict__ C,
                         int M, int K, int N) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ─── Host wrapper ─────────────────────────────────────────────────────────────

/**
 * Launch the naive CUDA matmul.
 *
 * @param h_A        Host input matrix A (M×K, row-major)
 * @param h_B        Host input matrix B (K×N, row-major)
 * @param h_C        Host output matrix C (M×N) — written on return
 * @param M, K, N    Dimensions
 * @param block_dim  Thread-block edge length (default 16)
 * @param kernel_ms  [out] Kernel execution time in milliseconds
 * @param transfer_ms [out] H2D + D2H transfer time in milliseconds
 */
void matmul_naive_cuda(const float* h_A, const float* h_B, float* h_C,
                       int M, int K, int N,
                       int block_dim,
                       double* kernel_ms, double* transfer_ms) {
    const size_t bytes_A = sizeof(float) * static_cast<size_t>(M) * K;
    const size_t bytes_B = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t bytes_C = sizeof(float) * static_cast<size_t>(M) * N;

    // ── Allocate device memory ────────────────────────────────────────────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes_A));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes_B));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes_C));

    // ── Time H2D transfers ────────────────────────────────────────────────
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

    // ── Launch kernel ─────────────────────────────────────────────────────
    dim3 block(block_dim, block_dim);
    dim3 grid((N + block_dim - 1) / block_dim,
              (M + block_dim - 1) / block_dim);

    CUDA_CHECK(cudaEventRecord(t0));
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(kerr));
        std::exit(EXIT_FAILURE);
    }

    float kern_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms_f, t0, t1));

    // ── D2H transfer ─────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, t0, t1));

    // ── Report ────────────────────────────────────────────────────────────
    if (kernel_ms)   *kernel_ms   = static_cast<double>(kern_ms_f);
    if (transfer_ms) *transfer_ms = static_cast<double>(h2d_ms + d2h_ms);

    // ── Cleanup ───────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
