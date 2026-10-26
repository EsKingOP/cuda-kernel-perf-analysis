#!/usr/bin/env bash
# profile_kernels.sh
#
# Automates Nsight Compute (ncu) and Nsight Systems (nsys) profiling of the
# matmul benchmark binary.
#
# Usage:
#   bash scripts/profile_kernels.sh [BINARY] [OUTPUT_DIR]
#
# Arguments:
#   BINARY      Path to the benchmark binary (default: ./build/benchmarks/bench_matmul)
#   OUTPUT_DIR  Directory for report files  (default: ./reports)
#
# Requirements:
#   - ncu  (Nsight Compute)   — typically at /usr/local/cuda/bin/ncu
#   - nsys (Nsight Systems)   — typically at /usr/local/cuda/bin/nsys
#   - Sudo / admin rights may be required for hardware counter access on Linux
#     (set perf_event_paranoid=0 or run with --target-processes all)
#
# Output files:
#   reports/ncu_<KERNEL>.ncu-rep    — Nsight Compute report (open in ncu-ui)
#   reports/ncu_export.csv          — CSV export of all metrics (for parse_nsight_csv.py)
#   reports/nsys_timeline.nsys-rep  — Nsight Systems timeline (open in nsys-ui)
#   reports/nsys_summary.txt        — Human-readable GPU summary

set -euo pipefail

BINARY="${1:-./build/benchmarks/bench_matmul}"
OUTPUT_DIR="${2:-./reports}"

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Binary not found: $BINARY"
    echo "Build first with: cmake --build build --parallel"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════"
echo " CUDA Kernel Profiling"
echo " Binary    : $BINARY"
echo " Output dir: $OUTPUT_DIR"
echo "════════════════════════════════════════"
echo ""

# ── 1. Nsight Compute: detailed kernel metrics ────────────────────────────────
#
# Sections collected:
#   SpeedOfLight       — roofline (compute + memory)
#   MemoryWorkloadAnalysis — DRAM / L2 / shared memory traffic
#   Occupancy          — achieved vs theoretical occupancy
#   SchedulerStats     — warp scheduler efficiency
#   WarpStateStats     — stall reasons
#   InstructionStats   — IPC, instruction mix
#
# --filter-name: profile only CUDA kernels matching "matmul"
# --target-processes all: profile child processes (useful for launchers)
# --set full: collect all available metrics (slower but complete)

echo "[1/3] Running Nsight Compute (full metric set)..."
ncu \
    --target-processes all \
    --filter-name "matmul" \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    --section InstructionStats \
    --csv \
    --log-file "${OUTPUT_DIR}/ncu_export.csv" \
    --export "${OUTPUT_DIR}/ncu_report" \
    "$BINARY" \
    2>&1 | tee "${OUTPUT_DIR}/ncu_run.log"

echo ""
echo "  ncu report : ${OUTPUT_DIR}/ncu_report.ncu-rep"
echo "  ncu CSV    : ${OUTPUT_DIR}/ncu_export.csv"
echo ""

# ── 2. Nsight Systems: application-level timeline ────────────────────────────
#
# Captures:
#   - CPU / GPU activity timeline
#   - CUDA API calls (cudaMalloc, cudaMemcpy, kernel launches)
#   - NVTX ranges (if any)
#   - GPU metrics sampled at 1 kHz
#
# Note: nsys generates a .nsys-rep file and a summary SQLite database.
# Open the .nsys-rep in nsys-ui (Nsight Systems GUI) for full timeline view.

echo "[2/3] Running Nsight Systems (timeline capture)..."
nsys profile \
    --trace=cuda,osrt,nvtx \
    --sample=cpu \
    --backtrace=dwarf \
    --output="${OUTPUT_DIR}/nsys_timeline" \
    --force-overwrite=true \
    --stats=true \
    "$BINARY" \
    > "${OUTPUT_DIR}/nsys_summary.txt" 2>&1

echo ""
echo "  nsys report : ${OUTPUT_DIR}/nsys_timeline.nsys-rep"
echo "  nsys summary: ${OUTPUT_DIR}/nsys_summary.txt"
echo ""

# ── 3. Quick roofline summary ────────────────────────────────────────────────

echo "[3/3] Extracting roofline metrics from ncu CSV..."
if command -v python3 &>/dev/null; then
    python3 scripts/parse_nsight_csv.py "${OUTPUT_DIR}/ncu_export.csv"
else
    echo "  (python3 not found — skipping parse step)"
fi

echo ""
echo "════════════════════════════════════════"
echo " Profiling complete."
echo " Open reports in:"
echo "   ncu-ui  ${OUTPUT_DIR}/ncu_report.ncu-rep"
echo "   nsys-ui ${OUTPUT_DIR}/nsys_timeline.nsys-rep"
echo "════════════════════════════════════════"
