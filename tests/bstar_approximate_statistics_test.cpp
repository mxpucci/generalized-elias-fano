#include <gtest/gtest.h>
#include "gef/B_STAR_GEF.hpp"
#include <bit>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

namespace {
template<typename T>
void check_approximation(const std::vector<T>& input, bool compare_serialization = false) {
    using U = std::make_unsigned_t<T>;
    using Codec = gef::internal::B_STAR_GEF<T>;
    if (input.empty()) return;
    const auto [minimum, maximum] = std::minmax_element(input.begin(), input.end());
    const U range = static_cast<U>(*maximum) - static_cast<U>(*minimum);
    const uint8_t width = std::bit_width(range);
    unsigned __int128 variation = static_cast<U>(static_cast<U>(input[0]) - static_cast<U>(*minimum));
    for (size_t i = 1; i < input.size(); ++i) {
        const U current = static_cast<U>(input[i]);
        const U previous = static_cast<U>(input[i - 1]);
        const U magnitude = input[i] >= input[i - 1] ? current - previous : previous - current;
        variation += magnitude;
    }
    const double estimate = variation ? std::clamp(
        std::log2(std::log(2.0) * static_cast<double>(variation) / input.size()), 0.0, double(width)) : 0.0;
    const uint8_t lower = static_cast<uint8_t>(std::floor(estimate));
    const uint8_t upper = static_cast<uint8_t>(std::ceil(estimate));
    const auto cost = [&](uint8_t split) {
        const auto bytes = [](size_t bits) { return (bits + 7) / 8; };
        size_t result = sizeof(T) + 2 + bytes(input.size() * split);
        if (split == width) return result;
        unsigned __int128 positive = 0, negative = 0;
        U previous = 0;
        for (const T value : input) {
            const U current = static_cast<U>(static_cast<U>(value) - static_cast<U>(*minimum)) >> split;
            if (current >= previous) positive += current - previous;
            else negative += previous - current;
            previous = current;
        }
        return result + bytes(static_cast<size_t>(positive) + input.size())
                      + bytes(static_cast<size_t>(negative) + input.size());
    };
    uint8_t expected = lower;
    size_t best_cost = cost(lower);
    for (const uint8_t candidate : {upper, width}) {
        if (cost(candidate) < best_cost) {
            best_cost = cost(candidate);
            expected = candidate;
        }
    }
    Codec actual(input, gef::APPROXIMATE_SPLIT_POINT);
    EXPECT_EQ(actual.split_point(), expected);
    EXPECT_EQ(actual.theoretical_size_in_bytes(), best_cost);
    std::vector<T> output(input.size() + 1, T(99));
    ASSERT_EQ(actual.get_elements_into(0, input.size(), output.data()), input.size());
    EXPECT_TRUE(std::equal(input.begin(), input.end(), output.begin()));
    EXPECT_EQ(output.back(), T(99));
    for (size_t i = 0; i < input.size(); ++i) EXPECT_EQ(actual[i], input[i]);
    if (compare_serialization) {
        Codec forced(input, gef::BlockParameters<T>{*minimum, width, expected});
        EXPECT_EQ(actual.size_in_bytes(), forced.size_in_bytes());
        const auto actual_path = std::filesystem::temp_directory_path() / "gef-bstar-approx-stats-actual.bin";
        const auto forced_path = std::filesystem::temp_directory_path() / "gef-bstar-approx-stats-forced.bin";
        actual.serialize(actual_path);
        forced.serialize(forced_path);
        std::ifstream actual_file(actual_path, std::ios::binary);
        std::ifstream forced_file(forced_path, std::ios::binary);
        const std::vector<char> actual_bytes((std::istreambuf_iterator<char>(actual_file)), {});
        const std::vector<char> forced_bytes((std::istreambuf_iterator<char>(forced_file)), {});
        EXPECT_EQ(actual_bytes, forced_bytes);
        std::filesystem::remove(actual_path);
        std::filesystem::remove(forced_path);
    }
}

template<typename T>
void check_type() {
    std::mt19937_64 random(719293);
    for (size_t length : {size_t(1), size_t(2), size_t(3), size_t(7), size_t(8), size_t(9),
                          size_t(63), size_t(64), size_t(65), size_t(257), size_t(32000)}) {
        std::vector<T> input(length, std::numeric_limits<T>::min());
        check_approximation(input, true);
        for (size_t i = 0; i < length; ++i)
            input[i] = i % 2 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        check_approximation(input, true);
        for (T& value : input) value = static_cast<T>(random());
        check_approximation(input, true);
        for (size_t i = 0; i < length; ++i) input[i] = static_cast<T>((i / 19) & 31);
        check_approximation(input, true);
    }
    // Small-block byte rounding and floor/ceil/raw tie cases.
    for (unsigned code = 0; code < 4096; ++code) {
        std::vector<T> input(6);
        for (unsigned i = 0; i < input.size(); ++i) input[i] = static_cast<T>((code >> (2 * i)) & 3);
        check_approximation(input);
    }
}
}

TEST(BStarApproximateStatistics, SignedAndUnsignedWidthsMatchSeparateDirectionalSums) {
    check_type<int8_t>();
    check_type<uint8_t>();
    check_type<int16_t>();
    check_type<uint16_t>();
    check_type<int32_t>();
    check_type<uint32_t>();
    check_type<int64_t>();
    check_type<uint64_t>();
}
