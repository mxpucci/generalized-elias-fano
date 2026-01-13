# Generalized Elias-Fano (GEF)

<p align="center">
  <b>The first compressed index to offer simultaneous state-of-the-art compression, constant-time random access, and high throughput for <i>arbitrary</i> integer sequences.</b>
</p>

<p align="center">
    <a href="#-performance">Performance</a>
    | <a href="#-quick-start">Quick Start</a>
    | <a href="#-variants">Variants</a>
    | <a href="#-citation">Paper</a>
</p>

<p align="center">
    <a href="https://github.com/mxpucci/generalized-elias-fano/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC--BY--NC--ND-blue" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/C++-23-blue.svg?style=flat&logo=c%2B%2B" alt="Language"></a>
</p>

---

## 🚀 Overview

**Generalized Elias-Fano (GEF)** is a family of lossless compression schemes that extends the theoretical and practical benefits of the classic Elias-Fano encoding to **non-monotone (unsorted)** data.

While traditional Elias-Fano is the gold standard for sorted arrays, it fails on arbitrary sequences. GEF bridges this gap. By analyzing the structural properties of runs and gaps in the high-order bits of integers, GEF achieves:

* **Universal Applicability:** Works efficiently on both monotonic *and* arbitrary integer sequences (time series, sensor logs, IDs).
* **Archival-Grade Compression:** Matches or beats general-purpose compressors like Brotli, Xz, and Zstd on many datasets.
* **In-Memory Speed:** Supports **constant-time random lookups** and decompression speeds exceeding **1 GB/s**.
* **Theoretical Guarantees:** Provides rigorous worst-case space bounds, with the $B^*$-GEF variant approaching the information-theoretic lower bound for Laplacian-distributed gaps.

## ⚡ Performance

GEF consistently sits on the **Pareto frontier** of compression ratio vs. access speed.

| Metric | vs. General Compressors (Xz, Brotli) | vs. Time-Series Compressors (ALP, LeCo) |
| :--- | :--- | :--- |
| **Space** | Comparable (often <1% difference) | **Up to 42% smaller** |
| **Throughput** | **Orders of magnitude faster** | Comparable (> 500 MB/s) |
| **Access** | **Constant-time Random Access** | Often requires block decompression |

## 📚 Classes Overview

The library exposes the following template classes in `<gef/gef.hpp>`:

1.  **RLE-GEF** (`gef::RLE_GEF`): Optimized for sequences with "runs" of values sharing high-order bits. Fastest random access.
2.  **U-GEF** (`gef::U_GEF` - Unidirectional): Adaptive compression for sequences with small positive gaps.
3.  **B-GEF** (`gef::B_GEF` - Bidirectional): Handles complex non-monotone data by encoding both positive and negative gaps.
4.  **B*-GEF** (`gef::B_STAR_GEF`): A simplified, symmetric variant that approaches the entropy lower bound for many real-world distributions.

### Performance Trade-offs

*   **Random Access**: The `random_access` template parameter (default: `true`) controls the index structure.
    *   When **disabled** (`false`), we do not build the auxiliary index for random access. This results in slightly less space usage and higher compression throughput. However, random access becomes slow because decompressing a specific element requires decompressing everything from the start of the partition.
*   **Split Point Strategy**:
    *   The classes support different split point strategies. Using an **approximated split point** (e.g., via `_APPROXIMATE` variants or constructor argument) yields higher compression throughput at the cost of slightly more space compared to the optimal strategy.


## 📋 Requirements

- **CMake 3.14+** for build configuration
- **C++20 compatible compiler** (GCC 10+, Clang 10+, MSVC 2019+)
- **Git** for dependency fetching

## 🔧 Installation & Integration

### Option 1: Git Submodule (Recommended)

```bash
# Add GEF as a submodule to your project
git submodule add https://github.com/mxpucci/generalized-elias-fano.git external/gef
git submodule update --init --recursive
```

```cmake
# In your CMakeLists.txt
add_subdirectory(external/gef)
target_link_libraries(your_target PRIVATE gef::gef)
```

### Option 2: CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    gef
    GIT_REPOSITORY https://github.com/mxpucci/generalized-elias-fano.git
    GIT_TAG        main  # Use specific version tag
)
FetchContent_MakeAvailable(gef)

target_link_libraries(your_target PRIVATE gef::gef)
```

### Option 3: Manual Download

```bash
git clone https://github.com/mxpucci/generalized-elias-fano.git
# Copy the gef directory to your project and use add_subdirectory()
```

## 💡 Quick Start

```cpp
#include <gef/gef.hpp>
#include <vector>
#include <iostream>

int main() {
    // Example with integer sequence compression
    std::vector<uint32_t> sequence = {1, 3, 7, 11, 15, 23, 31, 20, 10, 5}; // Can handle non-monotone
    
    // Create a B-GEF index (Bidirectional Generalized Elias-Fano)
    // Supports random access and non-monotone sequences
    gef::B_GEF<uint32_t> index(sequence);
    
    // Fast random access
    uint32_t value = index[3];  // Returns 11
    std::cout << "Value at index 3: " << value << std::endl;
    
    // Range queries (decompresses into a buffer)
    std::vector<uint32_t> buffer(5); // Buffer must be pre-allocated
    size_t count = index.get_elements(2, 5, buffer); // Get 5 elements starting at index 2
    
    std::cout << "Range [2, 7): ";
    for (auto val : buffer) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

## 🔨 Building from Source

### Building the Library

```bash
git clone https://github.com/mxpucci/generalized-elias-fano.git gef
cd gef
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running Tests

```bash
# From build directory
cmake -DGEF_BUILD_TESTS=ON ..
make -j$(nproc)
ctest --output-on-failure
```

### Running Benchmarks

```bash
# From build directory  
cmake -DGEF_BUILD_BENCHMARKS=ON ..
make -j$(nproc)
./benchmarks/simple_benchmark --benchmark_min_time=1s
```

## ⚙️ CMake Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `GEF_BUILD_TESTS` | Build the test suite | `ON` when main project |
| `GEF_BUILD_BENCHMARKS` | Build performance benchmarks | `ON` when main project |

## 📁 Project Structure

```
gef/
├── include/gef/          # Public API headers
├── tests/                # Comprehensive test suite
├── benchmarks/           # Performance benchmarks
├── examples/             # Usage examples and tutorials
├── cmake/                # CMake configuration files
└── CMakeLists.txt        # Main build configuration
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
