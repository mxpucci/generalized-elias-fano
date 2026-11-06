#include <gtest/gtest.h>
#include "gef/B_STAR_GEF.hpp"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

namespace {
template<typename T>
unsigned baseline_split(const std::vector<T>& values) {
    using U = std::make_unsigned_t<T>;
    const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
    const U range = U(U(*hi) - U(*lo));
    const unsigned width = std::bit_width(range);
    if (!width) return 0;
    unsigned __int128 variation = U(U(values[0]) - U(*lo));
    for (size_t i = 1; i < values.size(); ++i)
        variation += values[i] >= values[i - 1] ? U(U(values[i]) - U(values[i - 1]))
                                               : U(U(values[i - 1]) - U(values[i]));
    const double approx = std::clamp(std::log2(std::log(2.0) * double(variation) / values.size()), 0.0, double(width));
    const unsigned begin = unsigned(std::max(0.0, std::floor(approx) - 1));
    const unsigned end = unsigned(std::min(double(width), std::ceil(approx) + 3));
    const auto stats = total_variation_of_shifted_vec_with_multiple_shifts(
        values, *lo, *hi, uint8_t(begin), uint8_t(end), ExceptionRule::None);
    const auto cost = [&](const GapComputation& gaps, unsigned b) {
        const size_t n = gaps.positive_gaps + gaps.negative_gaps;
        size_t bytes = (n * b + 7) / 8;
        if (b < width) bytes += (gaps.sum_of_positive_gaps + n + 7) / 8 + (gaps.sum_of_negative_gaps + n + 7) / 8;
        return bytes;
    };
    unsigned best = begin;
    size_t best_cost = SIZE_MAX;
    for (unsigned b = begin; b <= end; ++b) {
        const size_t bytes = cost(stats[b - begin], b);
        if (bytes < best_cost) { best_cost = bytes; best = b; }
    }
    const size_t raw_cost = (values.size() * width + 7) / 8;
    if (raw_cost < best_cost) best = width;
    return best;
}

template<typename T>
void check(const std::vector<T>& values, bool serialization = false) {
    using U = std::make_unsigned_t<T>;
    using Codec = gef::internal::B_STAR_GEF<T>;
    const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
    const unsigned width = std::bit_width(U(U(*hi) - U(*lo)));
    const unsigned b = baseline_split(values);
    Codec codec(values, gef::OPTIMAL_SPLIT_POINT);
    EXPECT_EQ(codec.split_point(), b);
    Codec reference(values, gef::BlockParameters<T>{*lo, uint8_t(width), uint8_t(b)});
    EXPECT_EQ(codec.size_in_bytes(), reference.size_in_bytes());
    std::vector<T> decoded(values.size());
    ASSERT_EQ(codec.get_elements(0, values.size(), decoded), values.size());
    EXPECT_EQ(decoded, values);
    if (serialization) {
        const auto first = std::filesystem::temp_directory_path() / "gef-opt-stat-first.bin";
        const auto second = std::filesystem::temp_directory_path() / "gef-opt-stat-second.bin";
        codec.serialize(first);
        reference.serialize(second);
        std::ifstream a(first, std::ios::binary), z(second, std::ios::binary);
        EXPECT_EQ(std::vector<char>((std::istreambuf_iterator<char>(a)), {}),
                  std::vector<char>((std::istreambuf_iterator<char>(z)), {}));
        std::filesystem::remove(first);
        std::filesystem::remove(second);
    }
}

template<typename T>
void check_types() {
    using U = std::make_unsigned_t<T>;
    std::mt19937_64 rng(691715);
    for (unsigned width : {1u, 2u, 3u, 4u, 7u, 8u, 15u, 16u, 31u, 32u, 63u, 64u}) {
        if (width > sizeof(T) * 8) continue;
        const U mask = width == sizeof(T) * 8 ? U(~U(0)) : U((U(1) << width) - 1);
        for (size_t length : {size_t(2), size_t(7), size_t(8), size_t(9), size_t(63), size_t(64), size_t(65), size_t(257)}) {
            std::vector<T> values(length);
            for (auto& value : values) value = T(U(std::numeric_limits<T>::min()) + U(rng() & mask));
            values[0] = std::numeric_limits<T>::min();
            values[1] = T(U(values[0]) + mask);
            check(values, length == 65);
        }
    }
    check(std::vector<T>{std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), T(0), T(1)}, true);
    check(std::vector<T>{T(0)}, true);
}
}

TEST(BStarOptimalStatistics, DifferentialByteCostsAndAscendingTies) {
    for (unsigned length = 2; length <= 5; ++length) {
        const unsigned cases = 1u << (2 * length);
        for (unsigned word = 0; word < cases; ++word) {
            std::vector<int16_t> values(length);
            for (unsigned i = 0; i < length; ++i) values[i] = int16_t((word >> (2 * i)) & 3) - 2;
            check(values);
        }
    }
}
TEST(BStarOptimalStatistics, IntegralWidthsAndFullRange) {
    check_types<int8_t>(); check_types<uint8_t>();
    check_types<int16_t>(); check_types<uint16_t>();
    check_types<int32_t>(); check_types<uint32_t>();
    check_types<int64_t>(); check_types<uint64_t>();
}
