#include <benchmark/benchmark.h>
#include "gef/gef.hpp"

static void BM_Hello(benchmark::State& state) {
  for (auto _ : state) {
    gef::hello();
  }
}
BENCHMARK(BM_Hello);

BENCHMARK_MAIN(); 