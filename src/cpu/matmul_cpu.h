#pragma once
/**
 * matmul_cpu.h
 *
 * Declarations for CPU matrix-multiplication implementations.
 * All functions compute  C = A * B  where A is M×K, B is K×N, C is M×N.
 * Matrices are stored in row-major order as flat float arrays.
 */

/**
 * Naive triple-loop CPU matmul  (loop order: i, j, k).
 *
 * Access pattern for B (b[k*N + j]) is column-strided — cache-unfriendly
 * for large matrices.  Useful as a correctness reference.
 *
 * @param A   Input  M×K matrix (row-major)
 * @param B   Input  K×N matrix (row-major)
 * @param C   Output M×N matrix (row-major), zeroed by caller
 * @param M   Row count of A / C
 * @param K   Inner dimension
 * @param N   Column count of B / C
 */
void matmul_cpu_naive(const float* A, const float* B, float* C,
                      int M, int K, int N);

/**
 * Cache-friendly CPU matmul  (loop order: i, k, j — the "ikj" ordering).
 *
 * The innermost loop streams through a contiguous row of B and accumulates
 * into a contiguous row of C.  Both accesses are sequential, exploiting
 * L1/L2 cache lines.  Typically 4–8× faster than the naive ordering on
 * modern x86 hardware.
 *
 * @param A   Input  M×K matrix (row-major)
 * @param B   Input  K×N matrix (row-major)
 * @param C   Output M×N matrix (row-major), zeroed by caller
 * @param M   Row count of A / C
 * @param K   Inner dimension
 * @param N   Column count of B / C
 */
void matmul_cpu_cache_friendly(const float* A, const float* B, float* C,
                               int M, int K, int N);
