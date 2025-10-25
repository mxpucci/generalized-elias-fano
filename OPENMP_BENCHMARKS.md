# OpenMP Benchmark Variants

## Overview

The build system now generates **two versions** of the `compression_benchmark` executable to allow performance comparison between parallel and sequential decompression:

1. **`compression_benchmark`** - WITH OpenMP (parallel decompression)
2. **`compression_benchmark_no_omp`** - WITHOUT OpenMP (sequential decompression)

## Build Configuration

Both executables are built from the same source file (`benchmarks/compression_benchmark.cpp`) but with different compilation flags:

### compression_benchmark (WITH OpenMP)
- Compiled with OpenMP support enabled
- Uses `_OPENMP` preprocessor macro (defined by the compiler)
- `UniformedPartitioner::get_elements()` will use parallel decompression when spanning 3+ partitions
- Compile definition: `GEF_BENCHMARK_WITH_OPENMP=1`

### compression_benchmark_no_omp (WITHOUT OpenMP)
- Compiled with `-fno-openmp` flag
- `_OPENMP` macro is explicitly set to 0 to disable OpenMP code paths
- `UniformedPartitioner::get_elements()` always uses sequential decompression
- OpenMP pragmas are ignored (compiled with `-Wno-unknown-pragmas`)

## Building

Both executables are built automatically when you build the benchmarks:

```bash
cd build
cmake ..
make compression_benchmark compression_benchmark_no_omp -j$(nproc)
```

Or build everything:
```bash
make -j$(nproc)
```

## Usage

Both executables have identical command-line interfaces:

### Running All Benchmarks
```bash
# WITH OpenMP (parallel)
./build/benchmarks/compression_benchmark

# WITHOUT OpenMP (sequential)
./build/benchmarks/compression_benchmark_no_omp
```

### Running Specific Benchmarks
```bash
# Run only decompression throughput benchmarks
./build/benchmarks/compression_benchmark --benchmark_filter="Decompression"

# Run only B_GEF benchmarks
./build/benchmarks/compression_benchmark_no_omp --benchmark_filter="B_GEF"

# Run benchmarks for a specific file
./build/benchmarks/compression_benchmark --benchmark_filter="IT"
```

### Comparing Performance
```bash
# Run both and save results to JSON
./build/benchmarks/compression_benchmark \
    --benchmark_filter="Decompression" \
    --benchmark_out=results_with_omp.json \
    --benchmark_out_format=json

./build/benchmarks/compression_benchmark_no_omp \
    --benchmark_filter="Decompression" \
    --benchmark_out=results_no_omp.json \
    --benchmark_out_format=json

# Compare results using benchmark's compare.py tool
python3 -m pip install google-benchmark
compare.py benchmarks results_no_omp.json results_with_omp.json
```

## Decompression Throughput Benchmarks

The decompression throughput benchmarks measure how quickly each compressor can decompress an entire dataset using `get_elements(0, size)`. These benchmarks are available for:

- **B_GEF** (with and without RLE)
- **U_GEF**
- **RLE_GEF**

Each is tested with multiple `FIXED_PARTITION_SIZE` candidates:
- 512
- 1024
- 2048
- 4096
- 8192

And with different split point strategies:
- `APPROXIMATE_SPLIT_POINT`
- `OPTIMAL_SPLIT_POINT`

## Expected Performance Differences

### When OpenMP Helps (compression_benchmark)
- **Large partition counts**: When `get_elements()` spans 3+ partitions
- **Large datasets**: More data to decompress = more parallelization benefit
- **Multi-core systems**: Performance scales with available CPU cores

### When OpenMP Doesn't Help Much
- **Small ranges**: Single partition or 1-2 partition spans
- **Small datasets**: Parallelization overhead exceeds benefit
- **Single-core systems**: No parallelism available

## Implementation Details

The key difference is in `UniformedPartitioner::get_elements()` (lines 143-168 in `include/gef/UniformedPartitioner.hpp`):

```cpp
#ifdef _OPENMP
if (num_partitions_spanned >= 3) {
    // Parallel approach using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_partitions_spanned; ++i) {
        // Decompress partitions in parallel
    }
} else
#endif
{
    // Sequential fallback
}
```

When compiled **without OpenMP** (`compression_benchmark_no_omp`), the `#ifdef _OPENMP` block is skipped entirely, and only the sequential path is compiled.

## Technical Notes

### CMake Configuration

The special handling for `compression_benchmark` is in `benchmarks/CMakeLists.txt`:

- The benchmark is excluded from the generic `foreach(benchmark_file)` loop
- Two separate `add_executable()` calls are used
- For `compression_benchmark_no_omp`:
  - `-fno-openmp` disables OpenMP code generation
  - `_OPENMP=0` ensures the preprocessor doesn't see OpenMP as available
  - `-Wno-unknown-pragmas` suppresses warnings about unused `#pragma omp` directives

### Why Two Executables?

Having two separate executables allows you to:

1. **A/B test** OpenMP performance impact
2. **Deploy appropriately**: Use sequential version on single-core systems or in containers where OpenMP overhead isn't wanted
3. **Debug**: Easier to isolate OpenMP-related issues
4. **Benchmark fairly**: Compare identical code paths with only OpenMP as the variable

### Verification

To verify OpenMP is being used in the regular build, you can check for OpenMP symbols:

```bash
# Check if OpenMP is linked (macOS)
otool -L build/benchmarks/compression_benchmark | grep omp

# Check if OpenMP is linked (Linux)
ldd build/benchmarks/compression_benchmark | grep omp

# Or check if OpenMP symbols are present
nm build/benchmarks/compression_benchmark | grep -i omp
```

For the no-omp version, these checks should return nothing.

## Performance Monitoring

To monitor actual thread usage during benchmarks:

```bash
# macOS: Monitor thread count
while true; do ps -M $(pgrep compression_benchmark) | head -5; sleep 1; done

# Linux: Monitor thread count
watch -n 1 'ps -eLf | grep compression_benchmark | wc -l'
```

The OpenMP-enabled version should show multiple threads when processing large ranges across many partitions.

