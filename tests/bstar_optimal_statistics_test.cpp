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
std::pair<unsigned, unsigned> oracle_splits(const std::vector<T>& values) {
    using U = std::make_unsigned_t<T>;
    using Wide = unsigned __int128;
    const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
    const unsigned width = std::bit_width(U(U(*hi) - U(*lo)));
    if (!width) return {0, 0};
    Wide variation = U(U(values[0]) - U(*lo));
    for (size_t i = 1; i < values.size(); ++i)
        variation += values[i] >= values[i - 1] ? U(U(values[i]) - U(values[i - 1]))
                                               : U(U(values[i - 1]) - U(values[i]));
    const auto ceil_bytes = [](Wide bits) { return bits / 8 + (bits % 8 != 0); };
    const Wide n = values.size();
    std::vector<Wide> bits(width + 1), bytes(width + 1);
    for (unsigned b = 0; b < width; ++b) {
        U previous = 0;
        Wide positive = 0, negative = 0;
        for (const T value : values) {
            const U current = U(U(value) - U(*lo)) >> b;
            if (current >= previous) positive += current - previous;
            else negative += previous - current;
            previous = current;
        }
        bits[b] = n * (b + 2) + positive + negative;
        bytes[b] = ceil_bytes(n * b) + ceil_bytes(n + positive) + ceil_bytes(n + negative);
    }
    bits[width] = n * width;
    bytes[width] = ceil_bytes(bits[width]);
    const uint64_t mean = uint64_t(variation / n);
    const unsigned center = mean ? std::bit_width(mean) - 1 : 0;
    const unsigned begin = center ? center - 1 : 0;
    const unsigned end = std::min(width - 1, center + 2);
    unsigned best = begin;
    Wide best_bits = bits[width];
    for (unsigned b = begin; b <= end; ++b) {
        best_bits = std::min(best_bits, bits[b]);
        if (bytes[b] < bytes[best]) best = b;
    }
    if (bytes[width] < bytes[best]) best = width;
    // The literal neighborhood must contain an actual global unrounded optimum.
    EXPECT_EQ(uint64_t(best_bits), uint64_t(*std::min_element(bits.begin(), bits.end())));

    // Independently minimize the actual nominal byte objective over ALL splits.
    best = 0;
    for (unsigned b = 1; b <= width; ++b)
        if (bytes[b] < bytes[best]) best = b;

    // Independent original approximate objective: floor/ceil of ln(2)*g/n + raw.
    const double approx = std::clamp(std::log2(std::log(2.0) * double(variation) / values.size()), 0.0, double(width));
    const unsigned floor_b = unsigned(std::floor(approx)), ceil_b = unsigned(std::ceil(approx));
    unsigned approximate = floor_b;
    if (bytes[ceil_b] < bytes[approximate]) approximate = ceil_b;
    if (bytes[width] < bytes[approximate]) approximate = width;
    return {best, approximate};
}

template<typename T>
void check(const std::vector<T>& values, bool serialization = false) {
    using U = std::make_unsigned_t<T>;
    using Codec = gef::internal::B_STAR_GEF<T>;
    const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
    const unsigned width = std::bit_width(U(U(*hi) - U(*lo)));
    const auto [optimal, approximate] = oracle_splits(values);
    for (const auto [strategy, b] : {std::pair{gef::OPTIMAL_SPLIT_POINT, optimal},
                                     std::pair{gef::APPROXIMATE_SPLIT_POINT, approximate}}) {
        Codec codec(values, strategy);
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

TEST(BStarOptimalStatistics, PaperTheoremAndGlobalRoundedCosts) {
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

TEST(BStarOptimalStatistics, GlobalByteCertificationAndSparseOutliers) {
    check(std::vector<int16_t>{4, 0, 4}, true);
    check(std::vector<uint64_t>{632, 1947, 4505, 7041, 6898}, true);
    std::vector<int64_t> values(1000, -79);
    values[500] += int64_t(1) << 20;
    check(values, true);
    values.assign(257, -1);
    values[128] = 0;
    check(values, true);
}
