#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_path> [num_threads] [output_directory]"
    echo "  input_path: Can be a single .bin file or a directory containing .bin files"
    exit 1
fi

INPUT_PATH="$1"
NUM_THREADS="${2:-1}"
OUTPUT_DIR="${3:-./benchmark_results}"

mkdir -p "$OUTPUT_DIR"

# Collect files
FILES=()
if [ -f "$INPUT_PATH" ]; then
    FILES+=("$INPUT_PATH")
elif [ -d "$INPUT_PATH" ]; then
    # Use nullglob to handle case where no files match
    shopt -s nullglob
    FILES=("$INPUT_PATH"/*.bin)
    shopt -u nullglob
else
    echo "Error: '$INPUT_PATH' is not a valid file or directory"
    exit 1
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .bin files found in $INPUT_PATH"
    exit 1
fi

export OMP_NUM_THREADS=$NUM_THREADS

echo "Running benchmarks with OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Input: $INPUT_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Found ${#FILES[@]} file(s) to process."

BENCH_COMPRESSION="./build/benchmarks/compression_benchmark"
BENCH_RANDOM="./build/benchmarks/random_access_benchmark"
BENCH_DECOMP="./build/benchmarks/decompression_benchmark"

# Check if executables exist
if [ ! -f "$BENCH_COMPRESSION" ] || [ ! -f "$BENCH_RANDOM" ] || [ ! -f "$BENCH_DECOMP" ]; then
    echo "Error: Benchmark executables not found in ./build/benchmarks/"
    echo "Please build them first (e.g., cd build && cmake .. && make)"
    exit 1
fi

for FILE in "${FILES[@]}"; do
    BASENAME=$(basename "$FILE" .bin)
    echo "==================================================="
    echo "Processing: $BASENAME"
    echo "File: $FILE"
    echo "==================================================="

    # Run Compression Benchmark
    echo "  -> Running Compression Benchmarks..."
    "$BENCH_COMPRESSION" "$FILE" \
        --benchmark_out="$OUTPUT_DIR/${BASENAME}_compression.json" \
        --benchmark_out_format=json

    # Run Random Access Benchmark
    echo "  -> Running Random Access Benchmarks..."
    "$BENCH_RANDOM" "$FILE" \
        --benchmark_out="$OUTPUT_DIR/${BASENAME}_random_access.json" \
        --benchmark_out_format=json

    # Run Decompression Benchmark
    echo "  -> Running Decompression Benchmarks..."
    "$BENCH_DECOMP" "$FILE" \
        --benchmark_out="$OUTPUT_DIR/${BASENAME}_decompression.json" \
        --benchmark_out_format=json
        
    echo ""
done

echo "---------------------------------------------------"
echo "All benchmarks completed. Results saved to $OUTPUT_DIR"
