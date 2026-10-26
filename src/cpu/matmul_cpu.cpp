/**
 * matmul_cpu.cpp
 *
 * CPU implementations of general matrix multiplication (GEMM):
 *   C = A * B,   A: M×K,  B: K×N,  C: M×N  (row-major storage)
 *
 * Two variants:
 *   1. Naive  (ijk) — simple reference, cache-unfriendly for B
 *   2. Cache-friendly (ikj) — inner loop streams contiguous rows
 *
 * Neither uses SIMD intrinsics or threading; the purpose is to provide a
 * simple, auditable CPU baseline for correctness checking and to highlight
 * how badly the naive loop ordering performs relative to even a trivial
 * reordering.
 */

#include "matmul_cpu.h"
#include <cstring>

// ─── Naive (ijk) ──────────────────────────────────────────────────────────────

void matmul_cpu_naive(const float* __restrict__ A,
                      const float* __restrict__ B,
                            float* __restrict__ C,
                      int M, int K, int N) {
    // Zero output — caller may also do this, but be safe.
    std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                // A[i,k] is sequential along k; B[k,j] strides by N — cache miss.
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

// ─── Cache-friendly (ikj) ────────────────────────────────────────────────────

void matmul_cpu_cache_friendly(const float* __restrict__ A,
                                const float* __restrict__ B,
                                      float* __restrict__ C,
                                int M, int K, int N) {
    std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            // A[i,k] is a scalar broadcast across the entire inner loop.
            const float a_ik = A[i * K + k];
            // Inner loop: both B[k, 0..N-1] and C[i, 0..N-1] are contiguous
            // — compiler can auto-vectorise (AVX/SSE) and prefetcher hits well.
            const float* b_row = B + k * N;
                  float* c_row = C + i * N;
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}
