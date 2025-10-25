# Benchmark Hang Fix Summary

## Problem
The `compression_benchmark` was hanging during execution, preventing benchmarks from completing.

## Root Causes

### 1. Removed `--bitvector=both` Option
**Issue**: The `--bitvector=both` option was causing benchmark registration conflicts:
- Benchmarks were registered twice (once for SDSL, once for SUX)
- The global `g_factory` variable would point to whichever implementation was set last
- All benchmarks would use the wrong factory
- Duplicate registration caused undefined behavior and hangs

**Fix**: Removed the `--bitvector=both` option entirely. Users should now run benchmarks separately:
```bash
# Run with SDSL
./benchmarks/compression_benchmark --bitvector=sdsl data/file.bin

# Run with SUX
./benchmarks/compression_benchmark --bitvector=sux data/file.bin
```

### 2. Serialization Benchmark Missing Loop
**Issue**: The `SerializationSpaceBenchmark` function was not wrapping its code in the required `for (auto _ : state)` loop:
```cpp
// BROKEN - code outside the loop
void SerializationSpaceBenchmark(...) {
    compressor.serialize(temp_path);  // Not in loop!
}
```

This caused Google Benchmark to hang waiting for the benchmark to actually run.

**Fix**: Wrapped all benchmark code in the proper loop:
```cpp
// FIXED - code inside the loop
void SerializationSpaceBenchmark(...) {
    for (auto _ : state) {  // Required by Google Benchmark
        compressor.serialize(temp_path);
    }
}
```

### 3. Serialization Benchmarks Too Slow
**Issue**: Serialization benchmarks were running many iterations (default behavior), but we only need to measure file size once, not time performance.

**Fix**: Added `->Iterations(1)` to all serialization benchmark registrations:
```cpp
BENCHMARK_REGISTER_F(FileBasedCompressionBenchmark, B_GEF_Serialization_Space)
    ->Args({...})
    ->Iterations(1);  // Only run once to measure size
```

## Files Modified

### `benchmarks/compression_benchmark.cpp`
1. Removed `--bitvector=both` logic in `main()`
2. Fixed `SerializationSpaceBenchmark()` to wrap code in `for (auto _ : state)` loop
3. Added `->Iterations(1)` to all serialization benchmark registrations
4. Simplified argument parsing to only support `--bitvector=sdsl` or `--bitvector=sux`

### `SUX_BITVECTOR_IMPLEMENTATION.md`
- Updated documentation to reflect removal of `--bitvector=both`
- Added instructions for running benchmarks separately

## Verification

All benchmarks now complete successfully:

```bash
# Test compression benchmarks
./build/benchmarks/compression_benchmark data/BP.bin --bitvector=sux

# Test serialization benchmarks  
./build/benchmarks/compression_benchmark data/BP.bin --bitvector=sux --benchmark_filter="Serialization"
```

Output shows:
- ✅ All compression benchmarks complete
- ✅ All lookup benchmarks complete
- ✅ All serialization benchmarks complete with `iterations:1`
- ✅ Correct `serialized_size_in_bytes` and `serialized_bpi` metrics reported

## Lessons Learned

1. **Google Benchmark requires `for (auto _ : state)` loop**: All benchmark code must be inside this loop for proper execution
2. **Avoid global state in benchmarks**: The global `g_factory` variable caused issues when trying to support multiple implementations in one run
3. **Use `->Iterations(1)` for non-time benchmarks**: When measuring space/size rather than time, explicitly set iterations to 1
4. **Keep benchmark registration simple**: Registering the same benchmark multiple times with different global state is error-prone

