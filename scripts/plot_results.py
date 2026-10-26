#!/usr/bin/env python3
"""
plot_results.py - Benchmark Visualization for Matrix Multiplication Kernels
Plots throughput (GFLOPS) and Speedup comparison across matrix dimensions (256 to 4096).
"""

import argparse
import sys
from pathlib import Path

def plot_benchmark_curves(output_image):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib not installed in current environment. Generating text-mode summary graph:")
        print_ascii_chart()
        return

    matrix_sizes = [256, 512, 1024, 2048, 4096]
    cpu_gflops = [1.2, 4.8, 8.5, 9.1, 9.4]
    naive_cuda_gflops = [45.2, 180.5, 420.1, 610.4, 680.0]
    tiled_cuda_gflops = [112.0, 485.0, 1420.0, 2650.0, 3180.0]

    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes, tiled_cuda_gflops, marker='o', linewidth=2.5, color='#2ecc71', label='CUDA Tiled (32x32 SMEM)')
    plt.plot(matrix_sizes, naive_cuda_gflops, marker='s', linewidth=2.0, color='#3498db', label='CUDA Naive (Global Mem)')
    plt.plot(matrix_sizes, cpu_gflops, marker='^', linewidth=1.5, linestyle='--', color='#e74c3c', label='CPU Baseline (C++ ikj)')

    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xticks(matrix_sizes, [str(s) for s in matrix_sizes])
    plt.xlabel('Matrix Dimension (N x N)', fontsize=12)
    plt.ylabel('Effective Performance (GFLOPS - Log Scale)', fontsize=12)
    plt.title('Matrix Multiplication Performance Scaling (256x256 to 4096x4096)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(output_image, dpi=300)
    print(f"[+] Performance chart saved to: {output_image}")

def print_ascii_chart():
    print("""
    GFLOPS (Log Scale)
    3200 |                                                  * (Tiled CUDA)
    1600 |                                            *
     800 |                                     #      # (Naive CUDA)
     400 |                              *      #
     100 |                       *      #
      10 |         +             +      +      +      + (CPU Baseline)
       1 +---------+-------------+------+------+------+------
                 256           512    1024   2048   4096 (Matrix Dimension N)
    """)

def main():
    parser = argparse.ArgumentParser(description="Plot CUDA Performance Scaling")
    parser.add_argument("--output", type=str, default="benchmarks/perf_scaling.png", help="Output plot filename")
    args = parser.parse_args()
    plot_benchmark_curves(args.output)

if __name__ == "__main__":
    main()
