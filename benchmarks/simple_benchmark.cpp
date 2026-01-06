#include <benchmark/benchmark.h>
#include "gef/B_GEF.hpp"

static void BM_ConstructInternalBGEF(benchmark::State& state) {
  std::vector<int64_t> data;
  data.reserve(1000);
  for (int64_t i = 0; i < 1000; ++i) data.push_back(i);
  for (auto _ : state) {
    gef::internal::B_GEF<int64_t> bgef(data);
    benchmark::DoNotOptimize(bgef.size());
  }
}
BENCHMARK(BM_ConstructInternalBGEF);

BENCHMARK_MAIN(); 