#pragma once
/**
 * matrix_utils.h
 *
 * Host-side utilities for matrix allocation, initialisation, verification,
 * and wall-clock timing.  All matrices are stored in row-major order as
 * flat 1-D arrays of single-precision floats.
 */

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cassert>
#include <stdexcept>
#include <chrono>
#include <string>

// ─── Error-checking helpers ───────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ─── Matrix struct ────────────────────────────────────────────────────────────

struct Matrix {
    int    rows;
    int    cols;
    float* data;   ///< row-major, size = rows * cols

    /// Element access (host side)
    inline float& at(int r, int c)       { return data[r * cols + c]; }
    inline float  at(int r, int c) const { return data[r * cols + c]; }

    /// Total element count
    inline size_t numel() const { return static_cast<size_t>(rows) * cols; }

    /// Byte size
    inline size_t bytes() const { return numel() * sizeof(float); }
};

// ─── Allocation / deallocation ────────────────────────────────────────────────

/**
 * Allocate a host-pinned Matrix (rows × cols), zero-initialised.
 * Pinned memory enables async H2D/D2H transfers.
 */
inline Matrix matrix_alloc_pinned(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&m.data),
                              m.bytes()));
    std::memset(m.data, 0, m.bytes());
    return m;
}

/** Allocate a plain heap Matrix (rows × cols), zero-initialised. */
inline Matrix matrix_alloc(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = static_cast<float*>(std::calloc(m.numel(), sizeof(float)));
    if (!m.data) throw std::bad_alloc();
    return m;
}

/** Free pinned host memory. */
inline void matrix_free_pinned(Matrix& m) {
    if (m.data) { CUDA_CHECK(cudaFreeHost(m.data)); m.data = nullptr; }
}

/** Free heap host memory. */
inline void matrix_free(Matrix& m) {
    if (m.data) { std::free(m.data); m.data = nullptr; }
}

// ─── Initialisation ───────────────────────────────────────────────────────────

/**
 * Fill matrix with uniformly-distributed random floats in [lo, hi].
 * Uses a simple LCG seeded with the provided seed for reproducibility.
 */
inline void matrix_fill_random(Matrix& m, float lo = 0.0f, float hi = 1.0f,
                                unsigned seed = 42) {
    srand(seed);
    const float range = hi - lo;
    for (size_t i = 0; i < m.numel(); ++i) {
        m.data[i] = lo + range * (static_cast<float>(rand()) /
                                  static_cast<float>(RAND_MAX));
    }
}

/** Fill matrix with a constant value. */
inline void matrix_fill_const(Matrix& m, float val) {
    for (size_t i = 0; i < m.numel(); ++i) m.data[i] = val;
}

/** Fill identity matrix (must be square). */
inline void matrix_fill_identity(Matrix& m) {
    assert(m.rows == m.cols);
    std::memset(m.data, 0, m.bytes());
    for (int i = 0; i < m.rows; ++i) m.at(i, i) = 1.0f;
}

// ─── Verification ─────────────────────────────────────────────────────────────

/**
 * Element-wise comparison between two matrices.
 * Returns true if max absolute difference ≤ eps.
 * Prints the position and values of the first discrepancy found.
 */
inline bool matrix_verify(const Matrix& ref, const Matrix& got,
                           float eps = 1e-3f, bool verbose = false) {
    assert(ref.rows == got.rows && ref.cols == got.cols);
    float max_diff = 0.0f;
    int   err_r = -1, err_c = -1;

    for (int r = 0; r < ref.rows; ++r) {
        for (int c = 0; c < ref.cols; ++c) {
            float diff = std::fabs(ref.at(r, c) - got.at(r, c));
            if (diff > max_diff) {
                max_diff = diff;
                err_r = r; err_c = c;
            }
        }
    }

    bool ok = (max_diff <= eps);
    if (!ok || verbose) {
        fprintf(stderr,
                "[verify] max_diff=%.6f @ (%d,%d)  ref=%.6f  got=%.6f  %s\n",
                max_diff, err_r, err_c,
                (err_r >= 0 ? ref.at(err_r, err_c) : 0.0f),
                (err_r >= 0 ? got.at(err_r, err_c) : 0.0f),
                ok ? "PASS" : "FAIL");
    }
    return ok;
}

// ─── Timing ───────────────────────────────────────────────────────────────────

/** High-resolution wall-clock timer (milliseconds). */
struct WallTimer {
    using Clock = std::chrono::high_resolution_clock;
    using TP    = std::chrono::time_point<Clock>;

    TP t0;

    void start() { t0 = Clock::now(); }

    double elapsed_ms() const {
        auto dt = Clock::now() - t0;
        return std::chrono::duration<double, std::milli>(dt).count();
    }
};

// ─── GFLOP counting ──────────────────────────────────────────────────────────

/**
 * Returns the theoretical floating-point operation count for an M×K × K×N
 * matrix multiply: 2*M*K*N (one multiply + one add per element of the inner
 * product).
 */
inline double matmul_flops(int M, int K, int N) {
    return 2.0 * static_cast<double>(M) *
                 static_cast<double>(K) *
                 static_cast<double>(N);
}

/** Compute GFLOPS given flop count and elapsed milliseconds. */
inline double gflops(double flops, double ms) {
    return (ms > 0.0) ? (flops / ms) * 1e-6 : 0.0;
}
