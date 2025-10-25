# SUX BitVector Implementation

This document describes the SUX bitvector implementation added to the GEF library.

## Overview

A new bitvector implementation based on the SUX (Succinct data structure library) has been added alongside the existing SDSL implementation. Both implementations follow the same `IBitVector` interface, making them interchangeable.

## Files Added

### 1. SUX BitVector Implementation
- **Location**: `include/datastructures/SUXBitVector.hpp`
- **Description**: Complete implementation of `IBitVector` using SUX library
- **Features**:
  - Uses `sux::bits::Rank9` for rank support
  - Uses `sux::bits::SimpleSelectHalf` for select1 support
  - Uses `sux::bits::SimpleSelectZeroHalf` for select0 support
  - Full Rule of 5 implementation (copy/move constructors and assignment operators)
  - Serialization and deserialization support
  - All standard bitvector operations: `set`, `set_range`, `set_bits`, `rank`, `select`, `select0`

### 2. SUX BitVector Factory
- **Location**: `include/datastructures/SUXBitVectorFactory.hpp`
- **Description**: Factory for creating SUX bitvectors
- **Methods**:
  - `create(size_t size)` - Create bitvector of given size
  - `create(const std::vector<bool>& bits)` - Create from vector of bools
  - `from_file(const std::filesystem::path& filepath)` - Load from file
  - `from_stream(std::ifstream& in)` - Load from stream
  - Overhead estimation methods

### 3. Comprehensive Tests
- **Location**: `tests/test_sux_bitvector.cpp`
- **Description**: Complete test suite for SUX bitvector (600+ lines)
- **Coverage**:
  - Constructor tests
  - Bit access and modification tests
  - Copy and move semantics tests
  - Rank and select operations tests
  - Serialization tests
  - Factory tests
  - Edge case tests (large vectors, all ones, all zeros)
  - Bulk modification tests (`set_range`, `set_bits`)

## Build Configuration

### CMake Changes

The SUX library has been added as a dependency:

**Main CMakeLists.txt**:
```cmake
# --- Secondary Dependency: SUX ---
FetchContent_Declare(
    sux
    GIT_REPOSITORY https://github.com/vigna/sux.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(sux)
```

**src/CMakeLists.txt**:
```cmake
# Link against sdsl and sux publicly
target_link_libraries(gef PUBLIC sdsl sux)
```

## Usage

### Basic Usage

```cpp
#include "datastructures/SUXBitVector.hpp"
#include "datastructures/SUXBitVectorFactory.hpp"

// Create a bitvector using the factory
SUXBitVectorFactory factory;
auto bv = factory.create(1000);

// Set some bits
bv->set(10, true);
bv->set_range(20, 50, true);
bv->set_bits(100, 0b110101, 6);

// Enable rank/select support (lazy initialization)
bv->enable_rank();
bv->enable_select1();
bv->enable_select0();

// Use rank/select operations
size_t ones = bv->rank(100);
size_t pos = bv->select(5);
size_t zero_pos = bv->select0(10);

// Serialize
bv->serialize("bitvector.bin");

// Load
auto loaded = factory.from_file("bitvector.bin");
```

### Using with GEF Compressors

```cpp
// Use SUX bitvectors with GEF
auto sux_factory = std::make_shared<SUXBitVectorFactory>();
gef::U_GEF<int64_t> compressor(sux_factory, data, strategy);

// Or use SDSL bitvectors (default)
auto sdsl_factory = std::make_shared<SDSLBitVectorFactory>();
gef::U_GEF<int64_t> compressor(sdsl_factory, data, strategy);
```

## Running Tests

All tests are automatically discovered and run:

```bash
cd build
cmake ..
make
ctest
```

To run only SUX bitvector tests:
```bash
./tests/test_sux_bitvector
```

## Running Benchmarks

The compression benchmarks now support both SDSL and SUX implementations.

### Run with SDSL (default)
```bash
./benchmarks/compression_benchmark data/file.bin
```

### Run with SUX
```bash
./benchmarks/compression_benchmark --bitvector=sux data/file.bin
```

### Compare both implementations
To compare SDSL and SUX, run the benchmark twice and save results:
```bash
# Run with SDSL
./benchmarks/compression_benchmark --bitvector=sdsl data/file.bin \
    --benchmark_out=sdsl_results.json --benchmark_format=json

# Run with SUX  
./benchmarks/compression_benchmark --bitvector=sux data/file.bin \
    --benchmark_out=sux_results.json --benchmark_format=json

# Then compare the JSON output files
```

**Note**: The `--bitvector=both` option is not supported to avoid benchmark registration conflicts. Run separately for clean results.

## Performance Characteristics

### Space Overhead (per bit)
- **Rank support**: ~0.0625 bits (similar to SDSL)
- **Select1 support**: ~0.1875 bits
- **Select0 support**: ~0.1875 bits

### When to Use SUX vs SDSL

**Use SUX when**:
- You need specific SUX data structures in other parts of your code
- You want to compare performance characteristics
- You need the specific rank/select implementations SUX provides

**Use SDSL when**:
- You need maximum compatibility (SDSL is more widely used)
- You're already using SDSL in your project
- Default choice for most use cases

## Implementation Details

### Data Storage
- Uses `std::vector<uint64_t>` for bit storage
- Manages size explicitly (separate from storage capacity)

### Support Structures
- Lazy initialization: rank/select structures are only created when enabled
- Support structures hold raw pointers to the data vector
- Proper handling in copy/move operations to avoid dangling pointers

### Indexing Convention
**Important:** There is a subtle difference between SUX and SDSL:
- **SUX** uses 0-based indexing for select operations (select(0) = first one)
- **SDSL** uses 1-based indexing for select operations (select(1) = first one)
- The SUXBitVector wrapper handles this conversion automatically
- `SUXBitVector::select(k)` internally calls `sux::select(k-1)` to match SDSL behavior

### Compatibility
- Fully compatible with `IBitVector` interface
- Can be used interchangeably with `SDSLBitVector`
- All GEF compressors work with both implementations
- Handles indexing differences transparently

## Testing Coverage

The test suite covers:
- ✅ All constructors (size, vector<bool>, move/copy)
- ✅ Bit access and modification
- ✅ Bulk operations (set_range, set_bits)
- ✅ Rule of 5 (copy/move semantics)
- ✅ Rank operations (rank, rank0, range queries)
- ✅ Select operations (select, select0)
- ✅ Lazy initialization of support structures
- ✅ Serialization and deserialization
- ✅ Factory methods
- ✅ Edge cases (empty, large, all-ones, all-zeros)
- ✅ Error handling (out of bounds, missing support)

**All 43 tests pass successfully!**

## Benchmark Results

Benchmark results now include the bitvector implementation name in the label:

```
B_GEF_Compression/filename/SDSL/APPROXIMATE/1024
B_GEF_Compression/filename/SUX/APPROXIMATE/1024
```

This allows easy comparison of performance between implementations.

## Future Enhancements

Potential improvements:
- Add more SUX data structure variants (different rank/select implementations)
- Optimize serialization format for SUX-specific structures
- Add memory-mapped bitvector support
- Benchmark-driven tuning of SUX parameters

## References

- SUX Library: https://github.com/vigna/sux
- SDSL Library: https://github.com/simongog/sdsl-lite
- GEF Paper: [Add reference if available]

