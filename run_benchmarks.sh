#!/bin/bash

# =============================================================================
# Benchmark Runner Script
# =============================================================================
# This script runs both OpenMP-enabled and OpenMP-disabled compression benchmarks
# on all .bin files in a specified directory.
#
# Usage:
#   ./run_benchmarks.sh <input_directory> [output_directory]
#
# Arguments:
#   input_directory  - Directory containing .bin input files (required)
#   output_directory - Directory to save JSON results (optional, default: ./benchmark_results)
#
# Examples:
#   ./run_benchmarks.sh /path/to/data
#   ./run_benchmarks.sh /path/to/data ./my_results
# =============================================================================

# Exit immediately if a command fails
set -e

# --- Parse Arguments ---
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 <input_directory> [output_directory]"
    echo ""
    echo "Arguments:"
    echo "  input_directory   Directory containing .bin input files (required)"
    echo "  output_directory  Directory to save JSON results (optional, default: ./benchmark_results)"
    echo ""
    echo "Examples:"
    echo "  $0 /Users/michelangelopucci/Downloads/Data"
    echo "  $0 /path/to/data ./my_results"
    echo ""
    echo "This script runs both:"
    echo "  - compression_benchmark (WITH OpenMP - parallel decompression)"
    echo "  - compression_benchmark_no_omp (WITHOUT OpenMP - sequential decompression)"
    echo ""
    echo "Results are saved as separate JSON files for comparison."
    exit 0
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-./benchmark_results}"

# --- Configuration ---
BENCH_WITH_OMP="./build/benchmarks/compression_benchmark"
BENCH_NO_OMP="./build/benchmarks/compression_benchmark_no_omp"
# ---------------------

# --- Validation ---
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

if [ ! -f "$BENCH_WITH_OMP" ]; then
    echo "Error: Benchmark executable '$BENCH_WITH_OMP' not found."
    echo "Please build the benchmarks first with: cd build && make compression_benchmark"
    exit 1
fi

if [ ! -f "$BENCH_NO_OMP" ]; then
    echo "Error: Benchmark executable '$BENCH_NO_OMP' not found."
    echo "Please build the benchmarks first with: cd build && make compression_benchmark_no_omp"
    exit 1
fi

# Count .bin files
bin_count=$(find "$INPUT_DIR" -maxdepth 1 -name "*.bin" -type f | wc -l | tr -d ' ')
if [ "$bin_count" -eq 0 ]; then
    echo "Error: No .bin files found in '$INPUT_DIR'"
    exit 1
fi

# --- Setup ---
echo "========================================================================="
echo "Benchmark Configuration"
echo "========================================================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Files to process: $bin_count .bin file(s)"
echo "Benchmark variants:"
echo "  1. WITH OpenMP:    $BENCH_WITH_OMP"
echo "  2. WITHOUT OpenMP: $BENCH_NO_OMP"
echo "========================================================================="
echo ""

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# --- Run Benchmarks ---
file_counter=0

for input_file in "$INPUT_DIR"/*.bin; do
  # Check if the file exists to avoid errors with empty glob matches
  if [ -f "$input_file" ]; then
    file_counter=$((file_counter + 1))
    
    # Get the base name of the input file (e.g., "AP.bin" -> "AP")
    base_name=$(basename "$input_file" .bin)
    
    echo "[$file_counter/$bin_count] Processing '$base_name.bin'..."
    
    # --- Run WITH OpenMP ---
    output_with_omp="$OUTPUT_DIR/${base_name}_with_omp.json"
    echo "  -> Running WITH OpenMP (parallel)..."
    "$BENCH_WITH_OMP" "$input_file" \
        --benchmark_format=json \
        --benchmark_context=openmp=enabled \
        --benchmark_context=variant=with_omp \
        --benchmark_context=bitvector=sdsl \
        --benchmark_context=dataset="$base_name" \
        > "$output_with_omp" 2>&1
    echo "     Saved to: ${base_name}_with_omp.json"
    
    # --- Run WITHOUT OpenMP ---
    output_no_omp="$OUTPUT_DIR/${base_name}_no_omp.json"
    echo "  -> Running WITHOUT OpenMP (sequential)..."
    "$BENCH_NO_OMP" "$input_file" \
        --benchmark_format=json \
        --benchmark_context=openmp=disabled \
        --benchmark_context=variant=no_omp \
        --benchmark_context=bitvector=pasta \
        --benchmark_context=dataset="$base_name" \
        > "$output_no_omp" 2>&1
    echo "     Saved to: ${base_name}_no_omp.json"
    
    echo ""
  fi
done

echo "========================================================================="
echo "All benchmarks complete!"
echo "========================================================================="
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To compare performance between OpenMP and non-OpenMP versions:"
echo "  - Review JSON files in $OUTPUT_DIR"
echo "  - Files ending in '_with_omp.json' used parallel decompression"
echo "  - Files ending in '_no_omp.json' used sequential decompression"
echo ""
echo "You can use the Google Benchmark compare.py tool to generate comparisons:"
echo "  compare.py benchmarks ${OUTPUT_DIR}/<file>_no_omp.json ${OUTPUT_DIR}/<file>_with_omp.json"
echo "========================================================================="
