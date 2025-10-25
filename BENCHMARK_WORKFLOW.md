# Complete Benchmark Workflow Guide

## Overview

This guide provides a complete workflow for running, validating, and analyzing GEF compression benchmarks with OpenMP parallelization.

## Quick Start

```bash
# 1. Build both benchmark variants
cd build
cmake ..
make compression_benchmark compression_benchmark_no_omp -j$(nproc)
cd ..

# 2. Run benchmarks on all datasets
./run_benchmarks.sh /path/to/data

# 3. Validate results
./validate_benchmark_json.sh benchmark_results/AP_with_omp.json

# 4. Analyze results
cat benchmark_results/AP_with_omp.json | jq '.benchmarks[] | select(.name | contains("Decompression"))'
```

## Complete Workflow

### Step 1: Build Benchmarks

```bash
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGEF_BUILD_BENCHMARKS=ON
    
make compression_benchmark compression_benchmark_no_omp -j$(nproc)
cd ..
```

This creates two executables:
- `build/benchmarks/compression_benchmark` (WITH OpenMP)
- `build/benchmarks/compression_benchmark_no_omp` (WITHOUT OpenMP)

### Step 2: Run Benchmarks

#### Run All Datasets

```bash
./run_benchmarks.sh /Users/michelangelopucci/Downloads/Data
```

This will:
- Process all `.bin` files in the specified directory
- Run both OpenMP-enabled and OpenMP-disabled variants
- Save results to `benchmark_results/` with clear naming:
  - `<dataset>_with_omp.json`
  - `<dataset>_no_omp.json`

#### Run Single Dataset

```bash
# With OpenMP
./build/benchmarks/compression_benchmark data/AP.bin \
    --benchmark_format=json \
    --benchmark_context=openmp=enabled \
    --benchmark_context=dataset=AP \
    > results/AP_with_omp.json

# Without OpenMP
./build/benchmarks/compression_benchmark_no_omp data/AP.bin \
    --benchmark_format=json \
    --benchmark_context=openmp=disabled \
    --benchmark_context=dataset=AP \
    > results/AP_no_omp.json
```

### Step 3: Validate Results

```bash
# Validate a single JSON file
./validate_benchmark_json.sh benchmark_results/AP_with_omp.json

# Validate all results
for json_file in benchmark_results/*.json; do
    echo "Validating $json_file..."
    ./validate_benchmark_json.sh "$json_file"
done
```

The validation script checks:
- ✅ JSON validity
- ✅ Context metadata (OpenMP status, dataset name, etc.)
- ✅ All benchmark types present (Compression, Lookup, Serialization, Decompression)
- ✅ All compressor variants (B_GEF, B_GEF_NO_RLE, U_GEF, RLE_GEF)
- ✅ Custom counters (bpi, throughput, sizes)
- ✅ Timing data (real_time, cpu_time, iterations)
- ✅ No benchmark errors
- ✅ Expected partition sizes covered
- ✅ Reasonable benchmark counts

### Step 4: Analyze Results

#### Quick Analysis with jq

```bash
# Compare OpenMP vs no-OpenMP decompression throughput
echo "=== B_GEF Decompression Throughput (partition size 4096) ==="
echo -n "With OpenMP:    "
cat benchmark_results/AP_with_omp.json | jq '.benchmarks[] | select(.name | contains("B_GEF_Decompression") and contains("4096")) | .decompression_throughput_MBs' | head -1
echo -n "Without OpenMP: "
cat benchmark_results/AP_no_omp.json | jq '.benchmarks[] | select(.name | contains("B_GEF_Decompression") and contains("4096")) | .decompression_throughput_MBs' | head -1

# Find best compression ratio
echo "=== Best Compression Ratio ==="
cat benchmark_results/AP_with_omp.json | jq '[.benchmarks[] | select(.serialized_bpi)] | sort_by(.serialized_bpi) | first | {name, bpi: .serialized_bpi}'

# Find fastest compressor
echo "=== Fastest Compression ==="
cat benchmark_results/AP_with_omp.json | jq '[.benchmarks[] | select(.compression_throughput_MBs)] | sort_by(.compression_throughput_MBs) | reverse | first | {name, throughput: .compression_throughput_MBs}'
```

#### Detailed Analysis Script

Create a Python script for more sophisticated analysis:

```python
#!/usr/bin/env python3
import json
import sys

def analyze_benchmark(json_file):
    with open(json_file) as f:
        data = json.load(f)
    
    context = data['context']
    benchmarks = data['benchmarks']
    
    print(f"Dataset: {context.get('dataset', 'unknown')}")
    print(f"OpenMP: {context.get('openmp', 'unknown')}")
    print(f"Total benchmarks: {len(benchmarks)}")
    
    # Group by type
    by_type = {}
    for bench in benchmarks:
        bench_type = bench['name'].split('_')[-1].split('/')[0]
        by_type.setdefault(bench_type, []).append(bench)
    
    # Analyze decompression throughput
    if 'Decompression' in by_type:
        throughputs = [b['decompression_throughput_MBs'] 
                      for b in by_type['Decompression'] 
                      if 'decompression_throughput_MBs' in b]
        
        if throughputs:
            print(f"\nDecompression Throughput:")
            print(f"  Min: {min(throughputs):.2f} MB/s")
            print(f"  Max: {max(throughputs):.2f} MB/s")
            print(f"  Avg: {sum(throughputs)/len(throughputs):.2f} MB/s")
    
    # Analyze compression ratios
    if 'Space' in by_type:
        bpis = [b['serialized_bpi'] 
               for b in by_type['Space'] 
               if 'serialized_bpi' in b]
        
        if bpis:
            print(f"\nCompression Ratios (bits per integer):")
            print(f"  Best: {min(bpis):.2f} bpi")
            print(f"  Worst: {max(bpis):.2f} bpi")
            print(f"  Avg: {sum(bpis)/len(bpis):.2f} bpi")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <json_file>")
        sys.exit(1)
    
    analyze_benchmark(sys.argv[1])
```

### Step 5: Compare OpenMP Impact

Using Google Benchmark's `compare.py` tool:

```bash
# Install if needed
pip3 install google-benchmark

# Compare specific dataset
compare.py benchmarks \
    benchmark_results/AP_no_omp.json \
    benchmark_results/AP_with_omp.json

# Compare all datasets
for base_file in benchmark_results/*_no_omp.json; do
    base=$(basename "$base_file" _no_omp.json)
    echo "=== Comparing $base ==="
    compare.py benchmarks \
        "benchmark_results/${base}_no_omp.json" \
        "benchmark_results/${base}_with_omp.json" \
        | grep Decompression
    echo ""
done
```

### Step 6: Generate Publication Tables

Use the Python scripts to generate LaTeX tables:

```bash
python3 scripts/bench_tables.py benchmark_results/
```

This creates tables in `latex_tables/`:
- `table_compression_ratio.tex` - Compression ratios for all compressors
- `table_compression_throughput.tex` - Compression speeds
- `table_random_access.tex` - Lookup/query performance
- `table_decompression.tex` - Decompression throughput (showing OpenMP benefit)

## Data Captured

For each dataset and OpenMP variant, the benchmarks capture:

### Per Benchmark Type

| Type | Metrics | Use Case |
|------|---------|----------|
| **Compression** | time, throughput, size, bpi | How fast can we build the structure? |
| **Lookup** | time, throughput | How fast is random access (operator[])? |
| **Serialization Space** | size, bpi | How much disk space is needed? |
| **Decompression** | time, throughput | How fast can we decompress everything? |

### Compressor Coverage

- ✅ **B_GEF** (with RLE) - 56 benchmarks per variant
- ✅ **B_GEF_NO_RLE** (without RLE) - 56 benchmarks per variant
- ✅ **U_GEF** - 56 benchmarks per variant
- ✅ **RLE_GEF** - 28 benchmarks per variant

**Total**: 196 benchmarks × 2 variants (with/without OpenMP) = **392 benchmarks per dataset**

### Configuration Space

- **Partition sizes**: 512, 1024, 2048, 4096, 8192, 16384, 32768
- **Split strategies**: APPROXIMATE_SPLIT_POINT, OPTIMAL_SPLIT_POINT (where applicable)
- **OpenMP variants**: enabled, disabled

## Expected Results

### OpenMP Impact on Decompression

OpenMP parallelization is **most beneficial** when:

✅ **Large partition counts** (3+ partitions spanned by `get_elements`)
- Example: Decompressing entire dataset with small partition sizes (512, 1024)
- Expected speedup: 2-4x on 4-8 core systems

✅ **Large datasets** (millions of elements)
- More work to parallelize = better speedup
- Expected speedup: Scales with CPU cores

❌ **NOT beneficial** when:
- Single partition access
- Very small ranges (1-2 partitions)
- Small datasets where overhead exceeds benefit

### Typical Performance Metrics

Based on similar systems:

| Metric | Typical Range | Notes |
|--------|---------------|-------|
| Compression ratio (bpi) | 8-16 bits/integer | Dataset dependent |
| Compression throughput | 50-200 MB/s | Single-threaded |
| Lookup throughput | 100-500 MB/s | Cache-dependent |
| Decompression (no OMP) | 80-300 MB/s | Single-threaded |
| Decompression (with OMP) | 200-800 MB/s | Multi-threaded, 4-8 cores |

## Troubleshooting

### Benchmarks Too Slow

If benchmarks are taking too long:

```bash
# Run only decompression benchmarks
./build/benchmarks/compression_benchmark data/AP.bin \
    --benchmark_filter="Decompression" \
    --benchmark_format=json

# Use fewer partition size candidates
# (Edit compression_benchmark.cpp to remove some sizes)

# Reduce repetitions (if set)
--benchmark_repetitions=1
```

### Results Look Suspicious

If results seem inconsistent:

```bash
# Check system load
uptime

# Disable CPU frequency scaling (Linux)
sudo cpupower frequency-set --governor performance

# Disable Turbo Boost (macOS)
# (No easy command-line way, check in System Preferences)

# Run with more repetitions
--benchmark_repetitions=5 \
--benchmark_report_aggregates_only=true
```

### Missing Data in JSON

If the validation script reports missing data:

1. **Check benchmark compilation**: Ensure custom counters are set
2. **Check benchmark execution**: Look for error_message fields
3. **Verify benchmark registration**: Some compressors may have fewer variants

## File Reference

| File | Purpose |
|------|---------|
| `run_benchmarks.sh` | Main benchmark runner script |
| `validate_benchmark_json.sh` | JSON validation script |
| `BENCHMARK_DATA_GUIDE.md` | Complete data format documentation |
| `OPENMP_BENCHMARKS.md` | OpenMP variant details |
| `benchmarks/compression_benchmark.cpp` | Benchmark implementation |
| `scripts/bench_tables.py` | LaTeX table generator |

## Next Steps

After collecting benchmark data:

1. ✅ Validate all JSON files with `validate_benchmark_json.sh`
2. ✅ Compare OpenMP variants to quantify parallelization benefit
3. ✅ Identify best configurations (compression ratio vs speed tradeoff)
4. ✅ Generate publication-ready tables and figures
5. ✅ Document findings and optimal parameter choices

## Getting Help

For questions about:
- **Benchmark implementation**: See `benchmarks/compression_benchmark.cpp`
- **Data format**: See `BENCHMARK_DATA_GUIDE.md`
- **OpenMP variants**: See `OPENMP_BENCHMARKS.md`
- **Running benchmarks**: See `run_benchmarks.sh --help`
- **Validation**: See `validate_benchmark_json.sh --help`

