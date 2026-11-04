#include <gtest/gtest.h>
#include "gef/RLE_GEF.hpp"
#include <array>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>

namespace {
using Codec = gef::internal::RLE_GEF<int64_t>;

template<typename C>
void check_values(const C& codec, const std::vector<int64_t>& input) {
    ASSERT_EQ(codec.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i) ASSERT_EQ(codec[i], input[i]) << i;
    for (size_t start : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(127)}) {
        if (start >= input.size()) continue;
        for (size_t count : {size_t(1), size_t(7), size_t(64), size_t(129), input.size()}) {
            std::vector<int64_t> output(count + 1, 123456789);
            const size_t produced = codec.get_elements(start, count, output);
            ASSERT_EQ(produced, std::min(count, input.size() - start));
            for (size_t j = 0; j < produced; ++j) ASSERT_EQ(output[j], input[start + j]) << start << ',' << j;
            EXPECT_EQ(output[produced], 123456789);
            std::fill(output.begin(), output.end(), 123456789);
            ASSERT_EQ(codec.get_elements_into(start, count, output.data()), produced);
            for (size_t j = 0; j < produced; ++j) ASSERT_EQ(output[j], input[start + j]);
            EXPECT_EQ(output[produced], 123456789);
        }
    }
}
}

TEST(RLEEngineering, ConstantBlocksStoreOnlyTheBase) {
    for (const int64_t value : {int64_t(0), int64_t(-71), std::numeric_limits<int64_t>::max()}) {
        const std::vector<int64_t> input(1027, value);
        Codec codec(input);
        EXPECT_EQ(codec.b, 0);
        EXPECT_EQ(codec.h, 0);
        EXPECT_EQ(codec.L.size(), 0);
        EXPECT_EQ(codec.H.size(), 0);
        EXPECT_EQ(codec.B, nullptr);
        check_values(codec, input);
    }
}

TEST(RLEEngineering, ExactWidthsAndFullSignedDomain) {
    for (unsigned width = 1; width <= 63; ++width) {
        const int64_t max = static_cast<int64_t>((uint64_t(1) << width) - 1);
        const std::vector<int64_t> input{0, max, 0, max, 0, max};
        Codec codec(input);
        EXPECT_EQ(unsigned(codec.b) + codec.h, width);
        check_values(codec, input);
    }
    const std::vector<int64_t> input{
        std::numeric_limits<int64_t>::min(), 0, std::numeric_limits<int64_t>::max(), -1, 1};
    Codec codec(input);
    EXPECT_EQ(unsigned(codec.b) + codec.h, 64);
    check_values(codec, input);
}

TEST(RLEEngineering, EverySplitAndUnalignedRunBoundary) {
    std::mt19937_64 rng(52819);
    for (unsigned split = 0; split <= 64; ++split) {
        const unsigned width = std::min(64u, split + 5);
        const int64_t base = width >= 63 ? std::numeric_limits<int64_t>::min() : -1024;
        const uint64_t low_mask = split == 64 ? ~uint64_t(0) : ((uint64_t(1) << split) - 1);
        const uint64_t high_mask = split == 64 ? 0 : ((uint64_t(1) << (width - split)) - 1);
        std::vector<int64_t> input;
        for (size_t run_length : {size_t(1), size_t(2), size_t(63), size_t(64), size_t(65), size_t(127), size_t(11)}) {
            const uint64_t high = rng() & high_mask;
            for (size_t j = 0; j < run_length; ++j) {
                const uint64_t residual = (rng() & low_mask) | (split == 64 ? 0 : high << split);
                input.push_back(static_cast<int64_t>(static_cast<uint64_t>(base) + residual));
            }
        }
        Codec codec(input, gef::BlockParameters<int64_t>{base, uint8_t(width), uint8_t(split)});
        check_values(codec, input);
        Codec copied(codec);
        check_values(copied, input);
    }
}

TEST(RLEEngineering, SerializationOfConstantAndPackedBlocks) {
    const auto path = std::filesystem::temp_directory_path() / "gef-rle-engineering-test.bin";
    for (const std::vector<int64_t>& input : {
            std::vector<int64_t>(257, -15),
            std::vector<int64_t>{-8, -8, -7, 7, 7, -1, 0}}) {
        Codec codec(input);
        codec.serialize(path);
        Codec loaded;
        loaded.load(path);
        check_values(loaded, input);
        std::filesystem::remove(path);
    }
}

TEST(RLEEngineering, SupportFreePointerRanges) {
    std::vector<int64_t> input(771);
    for (size_t i = 0; i < input.size(); ++i) input[i] = int64_t((i / 37) * 1024 + i % 11) - 4000;
    gef::internal::RLE_GEF<int64_t, PastaRankBitVector, false> codec(input);
    check_values(codec, input);
    EXPECT_EQ(codec.get_elements_into(input.size(), 5, nullptr), 0);
    EXPECT_EQ(codec.get_elements_into(0, 0, nullptr), 0);
    std::vector<int64_t> short_output(1);
    EXPECT_THROW(codec.get_elements(1, 2, short_output), std::invalid_argument);
}
