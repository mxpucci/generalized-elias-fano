// Cost of the int64 <-> double conversion in the floating-point forks, on real data.
// For each dataset file (uint64 n, int64 decimals p, n int64 scaled values):
//   decode : integer decoder vs FP fork (bulk get_elements / get_doubles)
//   build  : integer constructor vs block-fill constructor copying integers vs FP fork
// Build a second binary with -DGEF_FP_IDENTITY_CONTROL: the FP fork then copies the
// int64 bits instead of converting, isolating the conversion from the decoder layout.
// Every FP result is checked bit for bit against the correctly rounded v / 10^p.
#include "gef/gef_fp.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

double strict_divide(int64_t v, double m); // fp_strict_divide.cpp (no fast-math)

namespace {
// The UniformPartitioning base of a GEF codec, to build it through the block-filling
// constructor without adding constructors to the integer classes.
template <class T, class C, size_t K, class... A>
gef::UniformPartitioning<T, C, K, A...> partitioning_base(const gef::UniformPartitioning<T, C, K, A...>*);
template <class I> using Base = decltype(partitioning_base(static_cast<const I*>(nullptr)));

template <class F> double best_of_5(F f) {
    double best = 1e9;
    for (int r = 0; r < 5; ++r) {
        const auto a = std::chrono::steady_clock::now(); f(); const auto b = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(b - a).count());
    }
    return best;
}

template <class I, class P, class... Args>
void run(const char* codec, const std::string& name, const std::vector<int64_t>& v, const std::vector<double>& x, int64_t p,
         Args... codec_args) {
    const size_t n = v.size();
    const double gb = n * 8 / 1e9;
    I integer(v);
    P fp(x.data(), n, p);
    std::vector<int64_t> ints(n, 1);
    std::vector<double> doubles(n, 1.0);
    fp.get_doubles(0, n, doubles.data());
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) mismatches += std::memcmp(&doubles[i], &x[i], 8) != 0;
    const double decode_int = best_of_5([&] { integer.get_elements(0, n, ints); });
    const double decode_fp = best_of_5([&] { fp.get_doubles(0, n, doubles.data()); });
    size_t sink = 0;
    const double build_int = best_of_5([&] { I c(v); sink += c.size_in_bytes(); });
    const double build_copy = best_of_5([&] {
        Base<I> c(typename Base<I>::from_blocks_t{}, n,
                  [&](size_t s, size_t l, int64_t* d) { std::memcpy(d, v.data() + s, l * 8); }, codec_args...);
        sink += c.size_in_bytes();
    });
    const double build_fp = best_of_5([&] { P c(x.data(), n, p); sink += c.size_in_bytes(); });
    std::printf("%s,%s,%zu,%s,%zu,%.3f,%.3f,%.3f,%.3f,%.3f\n", codec, name.c_str(), n,
                fp.size_in_bytes() - integer.size_in_bytes() <= (n / 32000 + 8) / 8 ? "equal+flags" : "DIFFER", mismatches,
                gb / decode_int, gb / decode_fp, gb / build_int, gb / build_copy, gb / build_fp + 0 * double(sink));
    std::fflush(stdout);
}
} // namespace

int main(int argc, char** argv) {
    std::printf("codec,dataset,n,sizes,fp_mismatches,decode_int_GBps,decode_fp_GBps,build_int_GBps,build_blockfill_copy_GBps,build_fp_GBps\n");
    for (int a = 1; a < argc; ++a) {
        FILE* f = std::fopen(argv[a], "rb");
        if (!f) { std::perror(argv[a]); return 1; }
        uint64_t n; int64_t p;
        if (std::fread(&n, 8, 1, f) != 1 || std::fread(&p, 8, 1, f) != 1) return 1;
        std::vector<int64_t> v(n);
        if (std::fread(v.data(), 8, n, f) != n) return 1;
        std::fclose(f);
        const double m = gef::fp::scale_for(p);
        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) x[i] = strict_divide(v[i], m);
        std::string name = argv[a];
        name = name.substr(name.find_last_of('/') + 1);
        name = name.substr(0, name.find('.'));
        run<gef::RLE_GEF<int64_t>, gef::RLE_GEF_FP<int64_t>>("RLE-GEF", name, v, x, p);
        run<gef::B_STAR_GEF_APPROXIMATE<int64_t>, gef::B_STAR_GEF_APPROXIMATE_FP<int64_t>>("B*-GEF", name, v, x, p,
                                                                                         gef::APPROXIMATE_SPLIT_POINT);
    }
}
