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

`B_STAR_GEF_APPROXIMATE` reads each block twice: one pass computes min, max and
the total variation `g`, and one pass encodes. It uses `b = max(0, floor(log2(g/n)))`,
the split point from the proof of the B*-GEF space bound, so every block takes
at most `n*(4 + log2(max(1, g/n)))` payload bits. It falls back to raw bit-packing
only when `n*W` is at most the payload lower bound `n*(b+2) + max(0, floor(g/2^b) - n)`.

### Variable partitions and cost oracles

`PARTITIONED_B_STAR_GEF` and `PARTITIONED_RLE_GEF` use the native block
compressors and the same access, decompression, size, and persistence interface.
They choose variable block boundaries with an approximate shortest-path algorithm.
Lookup uses a binary search over block endpoints: O(log(number of blocks)) with
random-access support enabled. The existing uniform variants are unchanged.

```cpp
gef::PartitionOptions options;
options.grid = 256;
options.epsilon = 0.1;
options.block_overhead_bits = 256;

gef::PARTITIONED_B_STAR_GEF<int64_t> exact_cost(values, options);
gef::PARTITIONED_B_STAR_GEF<int64_t, gef::BoundBStarOracle<int64_t>> fast_cost(values, options);
gef::PARTITIONED_RLE_GEF<int64_t> runs(values, options);

const auto& ends = exact_cost.partition_ends();
uint64_t modeled_bits = exact_cost.estimated_bits();
```

Here `values` is a `std::vector<int64_t>`. No prefix sum is applied.
The last template argument of either type can disable random-access support:
`PARTITIONED_B_STAR_GEF<T, Oracle, false>` or `PARTITIONED_RLE_GEF<T, false>`.

The grid permits cuts at multiples of `grid`, plus the final endpoint; use 1 to
consider every position. There is no maximum block size. Smaller grids cost more
construction time and temporary memory. `epsilon` controls pruning of the
partition graph, independently of the cost oracle. `block_overhead_bits` is a
positive, tunable charge per block, not a measurement of the native headers.

Let W be the width of the input after subtracting its minimum, and K the number
of permitted cuts. The built-in oracles have these costs:

| Oracle | Preprocessing | Temporary space | Interval query |
|--------|---------------|-----------------|----------------|
| `ExactBStarOracle<T>` | O(nW) | O(KW) | O(1), at most six splits plus raw |
| `BoundBStarOracle<T>` | O(n) | O(K) | O(1) |
| `RLEPartitionOracle<T>` | O(n + KW) | O(KW) | O(W) |

Widths are bounded by 64. RLE histogram entries currently reserve all 65 counters.
Oracles borrow the input vector; it must remain alive and unchanged while they
are used. The index retains only the encoded blocks and their directory.

For B*-GEF, define `y[i] = values[i] - min(values)` and
`G_b(l,r) = (y[l] >> b) + sum(abs((y[i] >> b) - (y[i-1] >> b)))`, where the sum
covers `l < i < r`. The initial high value is included because the native codec
starts its gap stream at zero. For m values the exact oracle minimizes
`m*(b+2) + G_b` over split points, also considering the codec's native raw case
`m*W`. It uses a constant-size candidate interval around
`floor(log2(ln(2)*G_0/m))`, with zero variation handled separately.

The fast oracle instead minimizes `m*(b+3) + ceil(G_0/2^b)` against `m*W` and
encodes with that selected split. This upper bound needs just one variation
prefix table. Its optimal cost exceeds the exact optimal payload by at most
2m bits. Both policies use a shared input minimum and width, including when
encoding: unlike uniform B*-GEF, they do not subtract each block's minimum.
This is a compression tradeoff that keeps interval costs cheap and monotone.

RLE uses prefix histograms of adjacent XOR widths. It removes each interval's
shared high prefix, then minimizes `m*(b+1) + runs_b*(W_block-b)` against raw
packing. Neither partitioned codec adds bitmap or all-ones fallbacks.

The planner keeps the farthest endpoint within geometrically spaced cost
thresholds, plus the next grid point. With a monotone oracle, its modeled cost
is within `1+epsilon` of the optimal partition on that grid. The fast B*-GEF
policy has this guarantee for its upper-bound objective. These are logical
payload bits plus the chosen block charge; they do not predict byte rounding
or the SDSL/Pasta rank/select and container overhead. Use `size_in_bytes()` to
compare the library-reported sizes of constructed indexes.

The planner uses O(K log_(1+epsilon)(Cmax/Cmin)) oracle calls, with
`Cmax = oracle(0,n).bits` and `Cmin = minimum_block_cost()`. This implementation
uses global thresholds; it does not implement the additional pruning needed
for the paper's tighter runtime bound.

The partitioner accepts any oracle exposing `size()`, `grid()`,
`minimum_block_cost()`, and `operator()(begin,end)` returning
`PartitionCost<T>{bits, BlockParameters<T>{base,width,split}}`. Its cost must not
increase when an interval is restricted, and must be at least the positive
`minimum_block_cost()`. Parameters must describe the supplied values and the
chosen native codec. To use an existing oracle object:

```cpp
gef::BoundBStarOracle<int64_t> oracle(values, options);
auto plan = gef::partition_sequence(oracle, options);
gef::PARTITIONED_B_STAR_GEF<int64_t, decltype(oracle)> index(values, oracle, options);
```

An existing oracle owns its block charge and grid. Rebuild it to change the
charge; passing a different `block_overhead_bits` to the planner does not alter
its costs. The planner checks that grids agree, and `epsilon` can change without
rebuilding the oracle.

`partition_sequence(oracle, options, true)` runs quadratic dynamic programming
on the same grid, useful for checking small instances. The reusable
`OptimalPartitioning<T, Compressor>` wrapper accepts the oracle object directly.
Native B*-GEF and RLE blocks now also accept explicit `BlockParameters<T>`;
ordinary constructors keep their existing normalization and split selection.
As with the existing native binary format, persistence assumes a compatible
build and trusted input. An individual B*-GEF unary gap must fit in 32 bits,
matching the native bulk decoder; explicit configurations exceeding this are
rejected before allocation.

### Floating-point data (`gef_fp.hpp`)

`RLE_GEF_FP` and `B_STAR_GEF_APPROXIMATE_FP` compress `double` values with a known number of decimal digits `p` and return `double`s. They store exactly the same structure as `RLE_GEF` and `B_STAR_GEF_APPROXIMATE` over the scaled integers `v = x · 10^p` (same size in bytes); the conversions are part of the codec:

- compression converts each 32,000-value partition into a reused buffer as it is built (`llrint(x · 10^p)`);
- decompression converts in registers before each value's only store; point and range access convert what they return;
- int → double is the correctly rounded `v / 10^p`, computed without a division (multiply by `10^-p`, one FMA correction) and written with intrinsics so that `-ffast-math` cannot change it.

```cpp
#include "gef/gef_fp.hpp"

std::vector<double> x = /* values with at most p decimal digits */;
gef::RLE_GEF_FP<int64_t> codec(x.data(), x.size(), /*p=*/2);
std::vector<double> out(x.size());
codec.get_doubles(0, x.size(), out.data());   // bit-identical to x
double y = codec.at_double(42);
```

`RLE_GEF_FP.hpp` and `B_STAR_GEF_FP.hpp` are generated from the integer headers by `scripts/generate_fp_forks.py` (the integer classes are not modified); regenerate them after changing the integer decoders. `benchmarks/bench_fp_conversion` measures the cost of the conversions on dataset files; `bench_fp_conversion_control` is the same program with the conversion replaced by a copy.

## Building

```bash
git clone https://github.com/mxpucci/generalized-elias-fano.git
cd generalized-elias-fano
mkdir build && cd build
cmake -DGEF_BUILD_TESTS=ON -DGEF_BUILD_BENCHMARKS=ON ..
make -j$(nproc)
ctest --output-on-failure
```

The standalone comparison executable can be enabled without Google Benchmark:

```bash
cmake -S . -B build -DGEF_BUILD_BENCHMARKS=OFF -DGEF_BUILD_PARTITION_BENCHMARK=ON
cmake --build build --target partition_oracle_comparison
./build/partition_oracle_comparison /path/to/DU.bin
```

It reads the full time-series binary file (64-bit count, optional 64-bit decimal
count, then signed 64-bit values), compares uniform and partitioned B*-GEF/RLE,
and checks every decoded value. Optional arguments set grid, epsilon, and block
overhead. Complete DU/CT results and reproduction details are in
[experiments/partition_oracles](experiments/partition_oracles/provenance.txt).

### Requirements

- CMake 3.14+
- C++20 compiler (GCC 10+, Clang 10+, MSVC 2019+)

## License

[Apache License 2.0](LICENSE)
