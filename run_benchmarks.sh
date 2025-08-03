#!/bin/bash

# --- Configuration ---
# 1. Set the path to your benchmark executable
BENCH_EXECUTABLE="./build/benchmarks/compression_benchmark"

# 2. Set the directory containing your .bin input files
INPUT_DIR="/Users/michelangelopucci/Downloads/Data"

# 3. Set the directory where you want to save the JSON results
OUTPUT_DIR="./benchmark_results"
# ---------------------

# Exit immediately if a command fails
set -e

# Create the output directory if it doesn't exist
echo "==> Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Find all .bin files in the input directory and loop through them
for input_file in "$INPUT_DIR"/*.bin; do
  # Check if the file exists to avoid errors with empty directories
  if [ -f "$input_file" ]; then
    # Get the base name of the input file (e.g., "AP.bin")
    base_name=$(basename "$input_file")
    
    # Create the output filename by replacing .bin with .json
    output_file="$OUTPUT_DIR/${base_name%.bin}.json"

    echo "==> Running benchmark on '$base_name'..."
    
    # Run the benchmark command for the single input file and redirect output
    "$BENCH_EXECUTABLE" "$input_file" --benchmark_format=json > "$output_file"
    
    echo "    ...Done. Results saved to '$output_file'"
  fi
done

echo "==> All benchmarks complete."
