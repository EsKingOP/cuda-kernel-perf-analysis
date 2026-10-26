#pragma once
/**
 * perf_metrics.h
 *
 * Structures and helpers for recording, aggregating, and exporting
 * performance measurements from matmul benchmark runs.
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <limits>

// ─── Per-run result ───────────────────────────────────────────────────────────

struct PerfResult {
    std::string impl_name;    ///< e.g. "cuda_tiled_32"
    int         matrix_size;  ///< N for an N×N square matrix
    int         run_index;    ///< 0-based

    double kernel_ms;         ///< kernel (or CPU compute) time in ms
    double transfer_ms;       ///< H2D + D2H cudaMemcpy time in ms (0 for CPU)
    double total_ms;          ///< kernel_ms + transfer_ms

    double gflops;            ///< effective GFLOP/s (based on 2*N^3 ops)
    double bandwidth_gb_s;    ///< estimated DRAM bandwidth (GB/s)

    bool   correct;           ///< numerical verification against reference
    int    tile_size;         ///< TILE_SIZE used (0 for non-tiled kernels)
};

// ─── Aggregate statistics ────────────────────────────────────────────────────

struct PerfStats {
    std::string impl_name;
    int         matrix_size;
    int         tile_size;

    double kernel_ms_mean;
    double kernel_ms_min;
    double kernel_ms_max;
    double kernel_ms_stddev;

    double gflops_mean;
    double gflops_max;

    double transfer_ms_mean;
    double bandwidth_gb_s_mean;
    bool   all_correct;
    int    num_runs;
};

inline PerfStats compute_stats(const std::vector<PerfResult>& runs) {
    PerfStats s{};
    if (runs.empty()) return s;

    s.impl_name   = runs[0].impl_name;
    s.matrix_size = runs[0].matrix_size;
    s.tile_size   = runs[0].tile_size;
    s.num_runs    = static_cast<int>(runs.size());
    s.all_correct = true;

    double sum_k = 0, sum_k2 = 0, sum_t = 0, sum_gf = 0, sum_bw = 0;
    double min_k =  std::numeric_limits<double>::max();
    double max_k = -std::numeric_limits<double>::max();
    double max_gf = 0;

    for (const auto& r : runs) {
        sum_k  += r.kernel_ms;
        sum_k2 += r.kernel_ms * r.kernel_ms;
        sum_t  += r.transfer_ms;
        sum_gf += r.gflops;
        sum_bw += r.bandwidth_gb_s;
        if (r.kernel_ms < min_k) min_k = r.kernel_ms;
        if (r.kernel_ms > max_k) max_k = r.kernel_ms;
        if (r.gflops    > max_gf) max_gf = r.gflops;
        if (!r.correct) s.all_correct = false;
    }

    double n = static_cast<double>(runs.size());
    s.kernel_ms_mean    = sum_k  / n;
    s.kernel_ms_min     = min_k;
    s.kernel_ms_max     = max_k;
    s.gflops_mean       = sum_gf / n;
    s.gflops_max        = max_gf;
    s.transfer_ms_mean  = sum_t  / n;
    s.bandwidth_gb_s_mean = sum_bw / n;

    double variance = (sum_k2 / n) - (s.kernel_ms_mean * s.kernel_ms_mean);
    s.kernel_ms_stddev = (variance > 0.0) ? std::sqrt(variance) : 0.0;

    return s;
}

// ─── CSV output ───────────────────────────────────────────────────────────────

/** Print CSV header to a file (or stdout). */
inline void csv_write_header(FILE* f) {
    fprintf(f,
        "impl,size,tile_size,run,"
        "kernel_ms,transfer_ms,total_ms,"
        "gflops,bandwidth_gb_s,correct\n");
}

/** Append a single PerfResult row to a CSV file. */
inline void csv_write_row(FILE* f, const PerfResult& r) {
    fprintf(f,
        "%s,%d,%d,%d,"
        "%.6f,%.6f,%.6f,"
        "%.2f,%.2f,%d\n",
        r.impl_name.c_str(),
        r.matrix_size,
        r.tile_size,
        r.run_index,
        r.kernel_ms,
        r.transfer_ms,
        r.total_ms,
        r.gflops,
        r.bandwidth_gb_s,
        r.correct ? 1 : 0);
}

/** Print a human-readable summary table to stdout. */
inline void print_stats_table(const std::vector<PerfStats>& stats_vec) {
    printf("\n%-20s %6s %5s %9s %9s %9s %9s %9s %s\n",
           "impl", "size", "tile",
           "mean_ms", "min_ms", "std_ms",
           "gflops", "bw_GB/s", "ok?");
    printf("%s\n", std::string(90, '-').c_str());
    for (const auto& s : stats_vec) {
        printf("%-20s %6d %5d %9.3f %9.3f %9.3f %9.1f %9.1f  %s\n",
               s.impl_name.c_str(),
               s.matrix_size,
               s.tile_size,
               s.kernel_ms_mean,
               s.kernel_ms_min,
               s.kernel_ms_stddev,
               s.gflops_mean,
               s.bandwidth_gb_s_mean,
               s.all_correct ? "PASS" : "FAIL");
    }
    printf("\n");
}
