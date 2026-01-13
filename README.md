# Generalized Elias Fano

Generalized Elias Fano (GEF) is a family of high-performance compressed indexes for both monotonically increasing and unsorted sequences, with theoretical guarantees.


## 📋 Requirements

- **CMake 3.14+** for build configuration
- **C++17 compatible compiler** (GCC 8+, Clang 7+, MSVC 2019+)
- **Git** for dependency fetching

## 🔧 Installation & Integration

### Option 1: Git Submodule (Recommended)

```bash
# Add GEF as a submodule to your project
git submodule add https://github.com/yourusername/gef.git external/gef
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
    GIT_REPOSITORY https://github.com/yourusername/gef.git
    GIT_TAG        v1.0.0  # Use specific version tag
)
FetchContent_MakeAvailable(gef)

target_link_libraries(your_target PRIVATE gef::gef)
```

### Option 3: Manual Download

```bash
git clone https://github.com/yourusername/gef.git
# Copy the gef directory to your project and use add_subdirectory()
```

## 💡 Quick Start

```cpp
#include <gef/gef.hpp>
#include <vector>
#include <iostream>

int main() {
    // Basic usage example
    gef::hello();
    
    // TODO: Add actual GEF API examples when implemented
    // Example with integer sequence compression:
    // std::vector<uint32_t> sequence = {1, 3, 7, 11, 15, 23, 31};
    // gef::EliasFano ef(sequence);
    // 
    // // Fast random access
    // uint32_t value = ef[3];  // Returns 11
    // 
    // // Range queries
    // auto range = ef.range(7, 23);  // Returns elements in [7, 23]
    
    return 0;
}
```

## 🔨 Building from Source

### Building the Library

```bash
git clone https://github.com/yourusername/gef.git
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
