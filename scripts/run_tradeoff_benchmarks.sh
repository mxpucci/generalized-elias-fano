#!/bin/bash

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <datasets_path> <threads>"
    exit 1
fi

DATASETS_PATH="$1"
THREADS="$2"
BINARY="./build/benchmarks/bench_partition_size_tradeoff"
OUTPUT_DIR="benchmark_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# List of compressors to benchmark
COMPRESSORS=(
    "RLE_GEF"
    "B_GEF"
    "B_GEF_APPROXIMATE"
    "B_STAR_GEF"
    "B_STAR_GEF_APPROXIMATE"
    "U_GEF"
    "U_GEF_APPROXIMATE"
)

echo "Starting benchmarks with ${THREADS} threads..."
echo "Datasets path: ${DATASETS_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

for COMP in "${COMPRESSORS[@]}"; do
    STRATEGY="OPTIMAL"
    if [[ "$COMP" == *"_APPROXIMATE" ]]; then
        STRATEGY="APPROXIMATE"
    fi
    
    OUTPUT_FILE="${OUTPUT_DIR}/${COMP}.json"
    
    echo "------------------------------------------------"
    echo "Running ${COMP} with strategy ${STRATEGY}..."
    
    "$BINARY" \
        "$DATASETS_PATH" \
        --compressor "$COMP" \
        --strategy "$STRATEGY" \
        --threads "$THREADS" \
        --random-access enabled \
        --benchmark_out="${OUTPUT_FILE}" \
        --benchmark_format=json
        
    if [ $? -eq 0 ]; then
        echo "Finished ${COMP}. Results saved to ${OUTPUT_FILE}"
    else
        echo "Error running ${COMP}"
    fi
done

echo "------------------------------------------------"
echo "All benchmarks completed."
