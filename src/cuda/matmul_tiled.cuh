#pragma once
/**
 * matmul_tiled.cuh
 *
 * Template declarations for the shared-memory tiled CUDA matmul kernel.
 *
 * The TILE_SIZE template parameter controls:
 *   - The edge length of the square tile loaded into __shared__ memory.
 *   - The thread-block dimensions: (TILE_SIZE × TILE_SIZE) threads per block.
 *   - Register pressure: larger tiles reuse data more but use more registers,
 *     which can reduce occupancy (see docs/profiling_analysis.md).
 *
 * Recommended values (Turing / Ampere):
 *   TILE_SIZE = 8  → low register pressure, high occupancy, low reuse
 *   TILE_SIZE = 16 → balanced; best occupancy/reuse trade-off on most GPUs
 *   TILE_SIZE = 32 → maximum reuse but often limited by register file; may
 *                    reduce occupancy on devices with 32K registers/SM
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ─── Forward declarations ─────────────────────────────────────────────────────

template <int TILE_SIZE>
__global__
void matmul_tiled_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                               float* __restrict__ C,
                         int M, int K, int N);

// ─── Host-callable wrapper ────────────────────────────────────────────────────

/**
 * Launch the tiled matmul kernel with the given TILE_SIZE.
 *
 * @param h_A         Host input A (M×K, row-major)
 * @param h_B         Host input B (K×N, row-major)
 * @param h_C         Host output C (M×N) — written on return
 * @param M, K, N     Matrix dimensions
 * @param tile_size   Must be 8, 16, or 32 (dispatched at compile time)
 * @param kernel_ms   [out] Kernel time in milliseconds
 * @param transfer_ms [out] H2D + D2H transfer time in milliseconds
 */
void matmul_tiled_cuda(const float* h_A, const float* h_B, float* h_C,
                       int M, int K, int N,
                       int tile_size,
                       double* kernel_ms, double* transfer_ms);
