#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>
#include "gef/B_STAR_GEF.hpp"

namespace {
template<class Codec, class T>
void check_all_accesses(const Codec& codec, const std::vector<T>& values) {
    std::vector<T> decoded(values.size() + 3, T(99));
    EXPECT_EQ(codec.get_elements(0, values.size(), decoded), values.size());
    EXPECT_TRUE(std::equal(values.begin(), values.end(), decoded.begin()));
    EXPECT_EQ(decoded[values.size()], T(99));
    for (size_t i = 0; i < values.size(); ++i) EXPECT_EQ(codec[i], values[i]);
    for (size_t start : {size_t(1), size_t(63), size_t(255), size_t(256), size_t(512)}) {
        if (start >= values.size()) continue;
        const size_t count = std::min<size_t>(257, values.size() - start);
        std::fill(decoded.begin(), decoded.end(), T(99));
        EXPECT_EQ(codec.get_elements_into(start, count, decoded.data()), count);
        EXPECT_TRUE(std::equal(values.begin() + start, values.begin() + start + count, decoded.begin()));
        EXPECT_EQ(decoded[count], T(99));
    }
    std::fill(decoded.begin(), decoded.end(), T(99));
    if (!values.empty()) {
        EXPECT_EQ(codec.get_elements_into(values.size() - 1, 3, decoded.data()), 1);
        EXPECT_EQ(decoded[0], values.back());
        EXPECT_EQ(decoded[1], T(99));
    }
    EXPECT_EQ(codec.get_elements_into(values.size(), 1, decoded.data()), 0);
    EXPECT_EQ(codec.get_elements_into(0, 0, nullptr), 0);
}
}

TEST(BStarEngineering, EverySplitAndUnalignedRange) {
    std::mt19937_64 random(793204);
    for (unsigned split = 0; split <= 64; ++split) {
        const unsigned width = std::min(64u, split + 10);
        const uint64_t low_mask = split == 64 ? UINT64_MAX : (uint64_t(1) << split) - 1;
        const uint64_t high_mask = (uint64_t(1) << (width - split)) - 1;
        for (size_t n : {size_t(1), size_t(63), size_t(64), size_t(65), size_t(257), size_t(513)}) {
            SCOPED_TRACE(::testing::Message() << "split=" << split << " n=" << n);
            std::vector<uint64_t> values(n);
            for (auto& value : values) {
                const uint64_t high = random() & high_mask;
                value = (random() & low_mask) | (split < 64 ? high << split : 0);
            }
            gef::BlockParameters<uint64_t> parameters{0, static_cast<uint8_t>(width), static_cast<uint8_t>(split)};
            gef::internal::B_STAR_GEF<uint64_t> codec(values, parameters);
            check_all_accesses(codec, values);
        }
    }
}

TEST(BStarEngineering, ConstantAndSingletonUseOnlyMetadata) {
    for (int64_t value : {INT64_MIN, int64_t(-1), int64_t(0), INT64_MAX}) {
        for (size_t n : {size_t(1), size_t(32000)}) {
            std::vector<int64_t> values(n, value);
            gef::internal::B_STAR_GEF<int64_t> codec(values);
            EXPECT_EQ(codec.split_point(), 0);
            EXPECT_LT(codec.size_in_bytes(), 64);
            check_all_accesses(codec, values);
            auto copy = codec;
            check_all_accesses(copy, values);
            auto moved = std::move(copy);
            check_all_accesses(moved, values);
        }
    }
}

TEST(BStarEngineering, FullWidthSignedAndUnsignedInputs) {
    const std::vector<int64_t> signed_values{INT64_MIN, INT64_MAX, 0, -1, INT64_MIN + 1, INT64_MAX - 1};
    const std::vector<uint64_t> unsigned_values{0, UINT64_MAX, 1, UINT64_MAX - 1, UINT64_MAX / 2};
    for (auto strategy : {gef::APPROXIMATE_SPLIT_POINT, gef::OPTIMAL_SPLIT_POINT}) {
        gef::internal::B_STAR_GEF<int64_t> signed_codec(signed_values, strategy);
        check_all_accesses(signed_codec, signed_values);
        gef::internal::B_STAR_GEF<uint64_t> unsigned_codec(unsigned_values, strategy);
        check_all_accesses(unsigned_codec, unsigned_values);
    }
}

TEST(BStarEngineering, SerializationAndBufferContract) {
    std::vector<int64_t> values(513);
    for (size_t i = 0; i < values.size(); ++i) values[i] = -1000 + (i % 5 ? i * 3 : i * 2);
    gef::internal::B_STAR_GEF<int64_t> codec(values);
    const auto path = std::filesystem::temp_directory_path() / "gef-bstar-engineering.bin";
    codec.serialize(path);
    gef::internal::B_STAR_GEF<int64_t> loaded;
    loaded.load(path);
    std::filesystem::remove(path);
    check_all_accesses(loaded, values);
    std::vector<int64_t> too_small(2);
    EXPECT_THROW(loaded.get_elements(0, 3, too_small), std::invalid_argument);
    EXPECT_EQ(loaded.get_elements(values.size(), 3, too_small), 0);
    auto copy = loaded;
    check_all_accesses(copy, values);
    auto moved = std::move(copy);
    check_all_accesses(moved, values);
}

TEST(BStarEngineering, NarrowSignedOffsetsCrossZero) {
    const std::vector<int8_t> values{127, -128, -1, 0, 1, 125, -120};
    for (auto strategy : {gef::APPROXIMATE_SPLIT_POINT, gef::OPTIMAL_SPLIT_POINT}) {
        gef::internal::B_STAR_GEF<int8_t> codec(values, strategy);
        check_all_accesses(codec, values);
    }
}

TEST(BStarEngineering, ApproximationStillComparesOnlyAdjacentCandidatesAndBitpacking) {
    std::mt19937_64 random(721993);
    for (size_t trial = 0; trial < 64; ++trial) {
        std::vector<int64_t> values(31 + trial * 3);
        int64_t value = -1000;
        for (auto& entry : values) {
            value += static_cast<int64_t>(random() % (trial + 1)) - static_cast<int64_t>(trial / 2);
            entry = value;
        }
        const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
        const uint8_t width = std::bit_width(static_cast<uint64_t>(*max_it - *min_it));
        uint64_t variation = values[0] - *min_it;
        for (size_t i = 1; i < values.size(); ++i)
            variation += values[i] > values[i - 1] ? values[i] - values[i - 1] : values[i - 1] - values[i];
        const double estimate = variation ? std::clamp(std::log2(std::log(2.0) * variation / values.size()), 0.0, double(width)) : 0.0;
        const uint8_t lower = static_cast<uint8_t>(std::floor(estimate));
        const uint8_t upper = static_cast<uint8_t>(std::ceil(estimate));
        uint8_t expected = lower;
        size_t best_bytes = SIZE_MAX;
        for (const uint8_t split : {lower, upper, width}) {
            gef::BlockParameters<int64_t> parameters{*min_it, width, split};
            gef::internal::B_STAR_GEF<int64_t> forced(values, parameters);
            if (forced.theoretical_size_in_bytes() < best_bytes) {
                expected = split;
                best_bytes = forced.theoretical_size_in_bytes();
            }
        }
        gef::internal::B_STAR_GEF<int64_t> codec(values);
        EXPECT_EQ(codec.split_point(), expected);
        check_all_accesses(codec, values);
    }
}
