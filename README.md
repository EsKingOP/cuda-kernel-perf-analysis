# CUDA Kernel Performance Analysis

A systematic benchmarking and profiling study of matrix multiplication on NVIDIA GPUs, comparing CPU baselines against naive CUDA and tiled (shared-memory) CUDA implementations across matrix sizes from 256×256 to 4096×4096.

---

## Overview

Matrix multiplication is the canonical compute-bound kernel for evaluating GPU performance. This project provides:

| Implementation | Description |
|---|---|
| **CPU Naive** | Triple-loop (`ijk`) reference — O(N³) scalar FMAs |
| **CPU Cache-Friendly** | Reordered loop (`ikj`) to exploit row-major locality |
| **CUDA Naive** | One thread per output element, global-memory-only |
| **CUDA Tiled** | Shared-memory tiling with configurable `TILE_SIZE` (8 / 16 / 32) |

### What is studied

- **Throughput scaling** — GFLOPS vs matrix size (256 → 4096)
- **Memory bandwidth pressure** — DRAM vs shared-memory traffic measured with Nsight Compute
- **Occupancy analysis** — how `TILE_SIZE` affects warp occupancy and register pressure
- **Transfer overhead** — cudaMemcpy host↔device timing isolated from kernel time
- **Numerical correctness** — tiled results verified against CPU reference (ε = 1e-3)
- **Repeated-run statistics** — 10 warm-up+timed runs per configuration to expose jitter

### Why single speedup numbers are misleading

At small sizes (256×256) the kernel launch overhead and H2D/D2H transfers can exceed compute time, making CUDA appear slower than CPU. At large sizes (4096×4096) the naive kernel becomes memory-bandwidth-limited while the tiled kernel achieves near-peak arithmetic throughput. The analysis document (`docs/profiling_analysis.md`) discusses this in depth.

---

## Directory Structure

```
cuda-kernel-perf-analysis/
├── CMakeLists.txt                  # Top-level build configuration
├── README.md
├── .gitignore
│
├── include/
│   ├── matrix_utils.h              # Matrix struct, alloc/fill/verify helpers
│   └── perf_metrics.h              # PerfResult struct, CSV logging
│
├── src/
│   ├── cpu/
│   │   ├── matmul_cpu.h            # CPU matmul declarations
│   │   └── matmul_cpu.cpp          # Naive (ijk) and cache-friendly (ikj) CPU impls
│   └── cuda/
│       ├── matmul_naive.cu         # Naive CUDA kernel (global memory)
│       ├── matmul_tiled.cuh        # Tiled kernel template header
│       └── matmul_tiled.cu         # Tiled CUDA kernel (shared memory)
│
├── benchmarks/
│   ├── bench_matmul.cu             # Full benchmark: all impls × all sizes, CSV output
│   └── bench_tile_sizes.cu         # Tile-size sweep at fixed 2048×2048
│
├── scripts/
│   ├── profile_kernels.sh          # ncu + nsys profiling automation
│   ├── parse_nsight_csv.py         # Parse ncu CSV export → summary table
│   └── plot_results.py             # Matplotlib speedup/throughput plots
│
└── docs/
    └── profiling_analysis.md       # Deep-dive: tiling, occupancy, bandwidth
```

---

## Prerequisites

| Tool | Minimum Version |
|---|---|
| CUDA Toolkit | 11.4+ |
| CMake | 3.20+ |
| GCC / MSVC | GCC 9+ / VS 2019+ |
| Nsight Compute (`ncu`) | 2021.2+ |
| Nsight Systems (`nsys`) | 2021.3+ |
| Python | 3.8+ (pandas, matplotlib) |

GPU targets: **sm_75** (Turing, e.g. RTX 20xx) and **sm_86** (Ampere, e.g. RTX 30xx).  
Edit `CMakeLists.txt` to add or change architectures.

---

## Build

```bash
# Configure (Release build)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile
cmake --build build --parallel

# Binaries produced:
#   build/benchmarks/bench_matmul
#   build/benchmarks/bench_tile_sizes
```

---

## Running the Benchmarks

### Full benchmark (all implementations × all sizes)

```bash
./build/benchmarks/bench_matmul
# Produces: results/bench_matmul.csv
```

Sample output:

```
impl,size,run,kernel_ms,transfer_ms,gflops,bandwidth_gb_s,correct
cpu_naive,256,1,5.21,0.00,6.43,0.00,1
cuda_naive,256,1,0.14,0.31,1202.3,142.1,1
cuda_tiled_16,256,1,0.09,0.31,1873.4,186.2,1
...
cuda_tiled_32,4096,10,28.47,89.11,4820.5,312.4,1
```

### Tile-size sweep

```bash
./build/benchmarks/bench_tile_sizes
# Produces: results/bench_tile_sizes.csv
```

---

## Profiling with Nsight

```bash
# Automated: runs both ncu and nsys, saves reports
bash scripts/profile_kernels.sh ./build/benchmarks/bench_matmul

# Parse Nsight Compute CSV export
python scripts/parse_nsight_csv.py reports/ncu_export.csv

# Plot results
python scripts/plot_results.py results/bench_matmul.csv
# Saves: results/speedup_curve.png, results/gflops_curve.png
```

---

## Key Results (RTX 3080 — Ampere sm_86)

| Matrix Size | CPU (ikj) GFLOPS | CUDA Naive GFLOPS | CUDA Tiled-32 GFLOPS | Speedup vs CPU |
|---|---|---|---|---|
| 256×256 | 14.2 | 892 | 1 580 | 111× |
| 512×512 | 13.8 | 2 310 | 4 120 | 298× |
| 1024×1024 | 12.1 | 3 870 | 6 940 | 573× |
| 2048×2048 | 11.4 | 4 120 | 7 890 | 692× |
| 4096×4096 | 10.9 | 3 980 | 8 210 | 753× |

> Numbers are illustrative; actual values depend on GPU, BIOS power limits, and thermal state.

---

## License

MIT — see source headers.
