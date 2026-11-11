#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "gef/gef_fp.hpp"

namespace {
// The correctly rounded double of v / 10^p, through the decimal string (strtod rounds
// correctly), so the expectation cannot be affected by -ffast-math in this test.
double decimal(int64_t v, int p) {
    const bool negative = v < 0;
    const uint64_t magnitude = negative ? uint64_t(0) - uint64_t(v) : uint64_t(v);
    uint64_t pow10 = 1;
    for (int i = 0; i < p; ++i) pow10 *= 10;
    char text[64];
    if (p == 0) std::snprintf(text, sizeof text, "%s%llu", negative ? "-" : "", (unsigned long long)magnitude);
    else std::snprintf(text, sizeof text, "%s%llu.%0*llu", negative ? "-" : "", (unsigned long long)(magnitude / pow10),
                       p, (unsigned long long)(magnitude % pow10));
    return std::strtod(text, nullptr);
}

// Scaled integers below 2^39 in magnitude with runs and random steps, as in the datasets.
std::vector<int64_t> series(size_t n, std::mt19937_64& rng) {
    std::vector<int64_t> v(n);
    std::uniform_int_distribution<int64_t> step(-5000, 5000), jump(-(int64_t(1) << 38), int64_t(1) << 38);
    std::bernoulli_distribution repeat(0.3), big(0.001);
    int64_t x = jump(rng) / 1024;
    for (auto& value : v) {
        if (big(rng)) x = jump(rng);
        else if (!repeat(rng)) x += step(rng);
        value = x;
    }
    return v;
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

template <class Int, class Fp>
void check(const std::vector<int64_t>& v, int p) {
    const size_t n = v.size();
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = decimal(v[i], p);
    Int integer(v);
    Fp fp(x.data(), n, p);
    ASSERT_EQ(fp.size(), n);
    // Same encoding as the integer codec, plus one bit per partition for the conversion choice.
    EXPECT_GE(fp.size_in_bytes(), integer.size_in_bytes());
    EXPECT_LE(fp.size_in_bytes(), integer.size_in_bytes() + (n / 32000 + 1 + 7) / 8);
    std::vector<double> decoded(n + 1, -1.0);
    ASSERT_EQ(fp.get_doubles(0, n, decoded.data()), n);
    for (size_t i = 0; i < n; ++i) ASSERT_TRUE(same_bits(decoded[i], x[i])) << "full decode, i=" << i << " p=" << p;
    for (size_t i = 0; i < n; i += 1 + n / 997) ASSERT_TRUE(same_bits(fp.at_double(i), x[i])) << "point, i=" << i;
    const size_t starts[] = {0, 1, 63, 64, 65, 127, 31999, 32000, 32001, n / 2, n - 65, n - 1};
    const size_t lengths[] = {1, 2, 63, 64, 65, 129, 1000, 32001, n};
    for (size_t start : starts) {
        if (start >= n) continue;
        for (size_t length : lengths) {
            const size_t count = std::min(length, n - start);
            ASSERT_EQ(fp.get_doubles(start, length, decoded.data()), count);
            for (size_t i = 0; i < count; ++i)
                ASSERT_TRUE(same_bits(decoded[i], x[start + i])) << "range start=" << start << " len=" << length;
        }
    }
}
} // namespace

TEST(FpFork, RleGefMatchesCorrectlyRoundedDecimals) {
    std::mt19937_64 rng(1);
    for (size_t n : {1, 63, 64, 65, 1000, 70001})
        for (int p : {0, 1, 2, 3, 5, 9}) check<gef::RLE_GEF<int64_t>, gef::RLE_GEF_FP<int64_t>>(series(n, rng), p);
}

TEST(FpFork, BStarGefMatchesCorrectlyRoundedDecimals) {
    std::mt19937_64 rng(2);
    for (size_t n : {1, 63, 64, 65, 1000, 70001})
        for (int p : {0, 1, 2, 3, 5, 9})
            check<gef::B_STAR_GEF_APPROXIMATE<int64_t>, gef::B_STAR_GEF_APPROXIMATE_FP<int64_t>>(series(n, rng), p);
}

TEST(FpFork, ConstantSequences) {
    for (int p : {0, 4}) {
        const std::vector<int64_t> v(70001, -123456789);
        check<gef::RLE_GEF<int64_t>, gef::RLE_GEF_FP<int64_t>>(v, p);
        check<gef::B_STAR_GEF_APPROXIMATE<int64_t>, gef::B_STAR_GEF_APPROXIMATE_FP<int64_t>>(v, p);
    }
}
