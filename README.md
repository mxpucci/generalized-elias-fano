# Generalized Elias-Fano (GEF)

<p align="center">
  <b>The first compressed index to offer simultaneous state-of-the-art compression, constant-time random access, and high throughput for <i>arbitrary</i> integer sequences.</b>
</p>

<p align="center">
  <a href="https://github.com/mxpucci/generalized-elias-fano/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

## Quick Start

```cpp
#include <gef/gef.hpp>

std::vector<uint32_t> data = {1, 3, 7, 11, 15, 23, 31, 20, 10, 5};

gef::B_GEF<uint32_t> index(data);

uint32_t val = index[3];  // O(1) random access

std::vector<uint32_t> buf(5);
index.get_elements(2, 5, buf);  // range decompression
```

## Installation

**CMake FetchContent:**

```cmake
include(FetchContent)
FetchContent_Declare(gef
    GIT_REPOSITORY https://github.com/mxpucci/generalized-elias-fano.git
    GIT_TAG main
)
FetchContent_MakeAvailable(gef)
target_link_libraries(your_target PRIVATE gef::gef)
```

**Submodule:**

```bash
git submodule add https://github.com/mxpucci/generalized-elias-fano.git external/gef
```

```cmake
add_subdirectory(external/gef)
target_link_libraries(your_target PRIVATE gef::gef)
```

## Variants

| Class | Use Case | Notes |
|-------|----------|-------|
| `gef::RLE_GEF<T>` | Sequences with runs of similar high-order bits | Fastest random access |
| `gef::U_GEF<T>` | Small positive gaps between elements | Unidirectional encoding |
| `gef::B_GEF<T>` | Complex non-monotone data | Bidirectional encoding |
| `gef::B_STAR_GEF<T>` | General purpose | Near-optimal for Laplacian gaps |

All variants support:
- `operator[]` — O(1) random access
- `get_elements(start, count, buffer)` — range decompression
- `size_in_bytes()` — compressed size
- `serialize()` / `load()` — persistence

### Template Parameters

```cpp
gef::B_GEF<T, partition_size, random_access>
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T` | — | Integer type (`int32_t`, `int64_t`, etc.) |
| `partition_size` | `32000` | Elements per partition |
| `random_access` | `true` | Set `false` to disable index (saves space, slower access) |

### Approximate Split Point

Use `*_APPROXIMATE` variants for faster compression at slight space cost:

```cpp
gef::B_GEF_APPROXIMATE<uint64_t> fast_index(data);
```

## Building

```bash
git clone https://github.com/mxpucci/generalized-elias-fano.git
cd generalized-elias-fano
mkdir build && cd build
cmake -DGEF_BUILD_TESTS=ON -DGEF_BUILD_BENCHMARKS=ON ..
make -j$(nproc)
ctest --output-on-failure
```

### Requirements

- CMake 3.14+
- C++20 compiler (GCC 10+, Clang 10+, MSVC 2019+)

## License

[Apache License 2.0](LICENSE)
