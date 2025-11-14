#!/bin/bash

# Measure compression throughput on a single dataset with OpenMP disabled,
# comparing SIMD-enabled and SIMD-disabled builds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

abs_path() {
    python3 - <<'PY' "$1"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

usage() {
    cat <<'EOF'
Usage: scripts/measure_throughput_no_omp.sh <dataset.bin> [--output-dir DIR] [--filter REGEX]

Arguments:
  <dataset.bin>        Path to the binary dataset to benchmark (required)

Options:
  --output-dir DIR     Directory to store JSON outputs (default: <repo>/bench-output)
  --filter REGEX       Google Benchmark filter (default: Compression$)

The script configures two separate build directories:
  - build/simd_on  (default SIMD optimizations)
  - build/simd_off (forces scalar fallbacks via GEF_DISABLE_SIMD)

For each build, compression_benchmark_no_omp is executed with OpenMP disabled.
Results are written as JSON files and a short throughput summary is printed.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

DATASET_ARG="$1"
shift

OUTPUT_DIR="${PROJECT_ROOT}/bench-output"
BENCH_FILTER="Compression$"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ $# -lt 2 ]] && { echo "Error: --output-dir requires a value" >&2; exit 1; }
            OUTPUT_DIR="$(abs_path "$2")"
            shift 2
            ;;
        --filter)
            [[ $# -lt 2 ]] && { echo "Error: --filter requires a value" >&2; exit 1; }
            BENCH_FILTER="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'" >&2
            usage
            exit 1
            ;;
    esac
done

DATASET_PATH="$(abs_path "${DATASET_ARG}")"
if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "Error: dataset '${DATASET_PATH}' not found" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

DATASET_BASENAME="$(basename "${DATASET_PATH}")"
DATASET_NAME="${DATASET_BASENAME%.*}"

configure_build() {
    local build_dir="$1"
    local disable_simd="$2"
    cmake -S "${PROJECT_ROOT}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGEF_BUILD_BENCHMARKS=ON \
        -DGEF_DISABLE_SIMD="${disable_simd}" >/dev/null
}

build_benchmark() {
    local build_dir="$1"
    cmake --build "${build_dir}" --target compression_benchmark_no_omp >/dev/null
}

run_benchmark() {
    local build_dir="$1"
    local simd_flag="$2"
    local output_path="$3"

    local bench_exec="${build_dir}/benchmarks/compression_benchmark_no_omp"
    if [[ ! -x "${bench_exec}" ]]; then
        echo "Error: benchmark executable '${bench_exec}' not found" >&2
        exit 1
    }

    echo "Running compression_benchmark_no_omp (simd=${simd_flag})..."
    "${bench_exec}" "${DATASET_PATH}" \
        --benchmark_filter="${BENCH_FILTER}" \
        --benchmark_format=json \
        --benchmark_out="${output_path}" \
        --benchmark_out_format=json \
        --benchmark_context=openmp=disabled \
        --benchmark_context=variant=no_omp \
        --benchmark_context=bitvector=sdsl \
        --benchmark_context=dataset="${DATASET_NAME}" \
        --benchmark_context=simd="${simd_flag}" \
        --benchmark_context=threads=1
}

print_summary() {
    local json_path="$1"
    python3 - "$json_path" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r") as fh:
    data = json.load(fh)
rows = []
for bench in data.get("benchmarks", []):
    value = bench.get("compression_throughput_MBs")
    if value is not None:
        rows.append((bench["name"], value))

if not rows:
    print("    No compression throughput counters found.")
    sys.exit(0)

rows.sort(key=lambda item: item[1], reverse=True)
print(f"    Top compression throughput results ({len(rows)} total):")
for name, value in rows[:5]:
    print(f"      {value:,.2f} MB/s\t{name}")
PY
}

echo "Dataset:        ${DATASET_PATH}"
echo "Output folder:  ${OUTPUT_DIR}"
echo "Benchmark filter: ${BENCH_FILTER}"
echo ""

SIMD_ON_BUILD="${PROJECT_ROOT}/build/simd_on"
SIMD_OFF_BUILD="${PROJECT_ROOT}/build/simd_off"

echo "Configuring SIMD-enabled build..."
configure_build "${SIMD_ON_BUILD}" OFF
build_benchmark "${SIMD_ON_BUILD}"

echo "Configuring SIMD-disabled build..."
configure_build "${SIMD_OFF_BUILD}" ON
build_benchmark "${SIMD_OFF_BUILD}"

SIMD_ON_OUTPUT="${OUTPUT_DIR}/${DATASET_NAME}_no_omp_simd_on.json"
SIMD_OFF_OUTPUT="${OUTPUT_DIR}/${DATASET_NAME}_no_omp_simd_off.json"

run_benchmark "${SIMD_ON_BUILD}" "on" "${SIMD_ON_OUTPUT}"
run_benchmark "${SIMD_OFF_BUILD}" "off" "${SIMD_OFF_OUTPUT}"

echo ""
echo "SIMD enabled results: ${SIMD_ON_OUTPUT}"
print_summary "${SIMD_ON_OUTPUT}"
echo ""
echo "SIMD disabled results: ${SIMD_OFF_OUTPUT}"
print_summary "${SIMD_OFF_OUTPUT}"
echo ""
echo "Done."


