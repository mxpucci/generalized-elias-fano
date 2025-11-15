#!/bin/bash

# =============================================================================
# Benchmark JSON Validation Script
# =============================================================================
# This script validates that benchmark JSON files contain all expected data
#
# Usage:
#   ./validate_benchmark_json.sh <json_file>
#
# Example:
#   ./validate_benchmark_json.sh benchmark_results/AP_with_omp.json
# =============================================================================

set -e

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 <json_file>"
    echo ""
    echo "Validates benchmark JSON output contains all expected fields and data."
    echo ""
    echo "Example:"
    echo "  $0 benchmark_results/AP_with_omp.json"
    exit 0
fi

JSON_FILE="$1"

if [ ! -f "$JSON_FILE" ]; then
    echo "Error: File '$JSON_FILE' not found"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed."
    echo "Install with: brew install jq (macOS) or apt-get install jq (Linux)"
    exit 1
fi

echo "========================================================================="
echo "Validating Benchmark JSON: $JSON_FILE"
echo "========================================================================="
echo ""

# Validation counters
errors=0
warnings=0

# Check 1: Valid JSON
echo "[1/10] Checking JSON validity..."
if jq empty "$JSON_FILE" 2>/dev/null; then
    echo "  ✅ Valid JSON"
else
    echo "  ❌ Invalid JSON format"
    ((errors++))
fi

# Check 2: Context section exists
echo "[2/10] Checking context section..."
if jq -e '.context' "$JSON_FILE" > /dev/null 2>&1; then
    echo "  ✅ Context section present"
    
    # Check custom context fields
    openmp=$(jq -r '.context.openmp // "missing"' "$JSON_FILE")
    variant=$(jq -r '.context.variant // "missing"' "$JSON_FILE")
    bitvector=$(jq -r '.context.bitvector // "missing"' "$JSON_FILE")
    dataset=$(jq -r '.context.dataset // "missing"' "$JSON_FILE")
    
    echo "     - OpenMP: $openmp"
    echo "     - Variant: $variant"
    echo "     - BitVector: $bitvector"
    echo "     - Dataset: $dataset"
    
    if [ "$openmp" = "missing" ] || [ "$variant" = "missing" ]; then
        echo "  ⚠️  Warning: Custom context fields may be missing"
        ((warnings++))
    fi
else
    echo "  ❌ Context section missing"
    ((errors++))
fi

# Check 3: Benchmarks array exists and not empty
echo "[3/10] Checking benchmarks array..."
if jq -e '.benchmarks | length > 0' "$JSON_FILE" > /dev/null 2>&1; then
    count=$(jq '.benchmarks | length' "$JSON_FILE")
    echo "  ✅ Found $count benchmark results"
else
    echo "  ❌ Benchmarks array missing or empty"
    ((errors++))
fi

# Check 4: Verify benchmark types
echo "[4/10] Checking benchmark types..."
expected_types=(
    "Compression"
    "Lookup"
    "Serialization_Space"
    "Decompression"
)

found_types=$(jq -r '[.benchmarks[].name | split("/")[0] | split("_")[-1]] | unique | .[]' "$JSON_FILE" | sort -u)

missing_types=()
for type in "${expected_types[@]}"; do
    if echo "$found_types" | grep -q "^${type}$"; then
        echo "  ✅ Found $type benchmarks"
    else
        echo "  ⚠️  Warning: No $type benchmarks found"
        missing_types+=("$type")
        ((warnings++))
    fi
done

# Check 5: Verify compressor variants
echo "[5/10] Checking compressor variants..."
expected_compressors=(
    "B_GEF"
    "B_GEF_NO_RLE"
    "U_GEF"
    "RLE_GEF"
)

found_compressors=$(jq -r '[.benchmarks[].name | split("/")[0] | sub("_Compression|_Lookup|_Serialization_Space|_Decompression"; "")] | unique | .[]' "$JSON_FILE" | sort -u)

for compressor in "${expected_compressors[@]}"; do
    if echo "$found_compressors" | grep -q "^${compressor}$"; then
        count=$(jq "[.benchmarks[].name | select(startswith(\"$compressor\"))] | length" "$JSON_FILE")
        echo "  ✅ Found $compressor benchmarks ($count results)"
    else
        echo "  ⚠️  Warning: No $compressor benchmarks found"
        ((warnings++))
    fi
done

# Check 6: Verify custom counters
echo "[6/10] Checking custom counters..."

# Check compression benchmarks have expected counters
compression_count=$(jq '[.benchmarks[] | select(.name | contains("Compression"))] | length' "$JSON_FILE")
if [ "$compression_count" -gt 0 ]; then
    compression_sample=$(jq '[.benchmarks[] | select(.name | contains("Compression"))] | first' "$JSON_FILE")
    
    has_size=$(echo "$compression_sample" | jq 'has("size_in_bytes")')
    has_bpi=$(echo "$compression_sample" | jq 'has("bpi")')
    has_throughput=$(echo "$compression_sample" | jq 'has("compression_throughput_MBs")')
    
    if [ "$has_size" = "true" ] && [ "$has_bpi" = "true" ] && [ "$has_throughput" = "true" ]; then
        echo "  ✅ Compression counters present (size_in_bytes, bpi, compression_throughput_MBs)"
    else
        echo "  ❌ Missing compression counters"
        ((errors++))
    fi
fi

# Check decompression benchmarks have expected counters
decompression_count=$(jq '[.benchmarks[] | select(.name | contains("Decompression"))] | length' "$JSON_FILE")
if [ "$decompression_count" -gt 0 ]; then
    decompression_sample=$(jq '[.benchmarks[] | select(.name | contains("Decompression"))] | first' "$JSON_FILE")
    
    has_throughput=$(echo "$decompression_sample" | jq 'has("decompression_throughput_MBs")')
    
    if [ "$has_throughput" = "true" ]; then
        echo "  ✅ Decompression counters present (decompression_throughput_MBs)"
    else
        echo "  ❌ Missing decompression counters"
        ((errors++))
    fi
fi

# Check size benchmarks have expected counters
size_count=$(jq '[.benchmarks[] | select(.name | contains("SizeInBytes"))] | length' "$JSON_FILE")
if [ "$size_count" -gt 0 ]; then
    size_sample=$(jq '[.benchmarks[] | select(.name | contains("SizeInBytes"))] | first' "$JSON_FILE")
    
    has_size=$(echo "$size_sample" | jq 'has("size_in_bytes")')
    has_bpi=$(echo "$size_sample" | jq 'has("bpi")')
    
    if [ "$has_size" = "true" ] && [ "$has_bpi" = "true" ]; then
        echo "  ✅ Size counters present (size_in_bytes, bpi)"
    else
        echo "  ❌ Missing size counters"
        ((errors++))
    fi
fi

# Check lookup benchmarks have expected counters
lookup_count=$(jq '[.benchmarks[] | select(.name | contains("Lookup"))] | length' "$JSON_FILE")
if [ "$lookup_count" -gt 0 ]; then
    lookup_sample=$(jq '[.benchmarks[] | select(.name | contains("Lookup"))] | first' "$JSON_FILE")
    
    has_throughput=$(echo "$lookup_sample" | jq 'has("lookup_throughput_MBs")')
    
    if [ "$has_throughput" = "true" ]; then
        echo "  ✅ Lookup counters present (lookup_throughput_MBs)"
    else
        echo "  ❌ Missing lookup counters"
        ((errors++))
    fi
fi

# Check 7: Verify timing data
echo "[7/10] Checking timing data..."
sample_bench=$(jq '.benchmarks[0]' "$JSON_FILE")

has_real_time=$(echo "$sample_bench" | jq 'has("real_time")')
has_cpu_time=$(echo "$sample_bench" | jq 'has("cpu_time")')
has_iterations=$(echo "$sample_bench" | jq 'has("iterations")')

if [ "$has_real_time" = "true" ] && [ "$has_cpu_time" = "true" ] && [ "$has_iterations" = "true" ]; then
    echo "  ✅ Timing data present (real_time, cpu_time, iterations)"
else
    echo "  ❌ Missing timing data"
    ((errors++))
fi

# Check 8: Check for error messages
echo "[8/10] Checking for benchmark errors..."
error_count=$(jq '[.benchmarks[] | select(.error_message)] | length' "$JSON_FILE")
if [ "$error_count" -eq 0 ]; then
    echo "  ✅ No benchmark errors"
else
    echo "  ⚠️  Warning: Found $error_count benchmark(s) with errors"
    jq -r '.benchmarks[] | select(.error_message) | "     - \(.name): \(.error_message)"' "$JSON_FILE"
    ((warnings++))
fi

# Check 9: Verify partition size coverage
echo "[9/10] Checking partition size coverage..."
expected_sizes=(512 1024 2048 4096 8192 16384 32768)
found_sizes=$(jq -r '[.benchmarks[].name | capture("/(?<size>[0-9]+)$").size] | unique | .[]' "$JSON_FILE" 2>/dev/null | sort -n)

missing_sizes=()
for size in "${expected_sizes[@]}"; do
    if echo "$found_sizes" | grep -q "^${size}$"; then
        count=$(jq "[.benchmarks[].name | select(endswith(\"/$size\"))] | length" "$JSON_FILE")
        echo "  ✅ Found partition size $size ($count results)"
    else
        echo "  ⚠️  Warning: No benchmarks with partition size $size"
        missing_sizes+=("$size")
        ((warnings++))
    fi
done

# Check 10: Summary statistics
echo "[10/10] Computing summary statistics..."
total_benchmarks=$(jq '.benchmarks | length' "$JSON_FILE")
total_time=$(jq '[.benchmarks[].real_time] | add' "$JSON_FILE")
total_time_sec=$(echo "scale=2; $total_time / 1000000000" | bc)

echo "  📊 Total benchmarks: $total_benchmarks"
echo "  📊 Total benchmark time: ${total_time_sec}s"

if [ "$total_benchmarks" -lt 150 ]; then
    echo "  ⚠️  Warning: Expected ~196 benchmarks per dataset, found $total_benchmarks"
    ((warnings++))
fi

# Final summary
echo ""
echo "========================================================================="
echo "Validation Summary"
echo "========================================================================="
if [ $errors -eq 0 ] && [ $warnings -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED - JSON file is valid and complete"
    exit 0
elif [ $errors -eq 0 ]; then
    echo "⚠️  PASSED WITH WARNINGS - $warnings warning(s) found"
    echo ""
    echo "The file is usable but may be missing some expected data."
    exit 0
else
    echo "❌ VALIDATION FAILED - $errors error(s), $warnings warning(s)"
    echo ""
    echo "The file has critical issues that should be investigated."
    exit 1
fi

