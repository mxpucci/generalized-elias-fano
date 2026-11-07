#include <gtest/gtest.h>
#include "gef/RLE_GEF.hpp"
#include <algorithm>
#include <limits>
#include <vector>

namespace {
template<typename T>
void check_packed_ranges() {
    constexpr unsigned bits = sizeof(T) * 8;
    using C = gef::internal::RLE_GEF<T>;
    std::vector<unsigned> splits;
    for (unsigned split = 0; split <= std::min(33u, bits); ++split)
        splits.push_back(split);
    for (unsigned split : {bits - 1, bits})
        if (std::find(splits.begin(), splits.end(), split) == splits.end())
            splits.push_back(split);

    for (unsigned split : splits) {
        const uint64_t low_mask = split == 64 ? ~uint64_t(0) : (uint64_t(1) << split) - 1;
        for (bool with_high : {false, true}) {
            if (with_high && split == bits) continue;
            const unsigned width = split + unsigned(with_high);
            const uint64_t residual_mask = width == 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
            // Both endpoints are valid bases for this residual width. The
            // unsigned subtraction also handles the full signed type domain.
            const T upper_base = static_cast<T>(uint64_t(std::numeric_limits<T>::max()) - residual_mask);
            for (T base : {std::numeric_limits<T>::min(), upper_base}) {
                SCOPED_TRACE(::testing::Message() << "split=" << split << " high=" << with_high
                                                 << " base=" << +base);
                std::vector<T> input(269);
                for (size_t i = 0; i < input.size(); ++i) {
                    uint64_t residual = (i * 0x9e3779b97f4a7c15ULL) & low_mask;
                    if (with_high && (i / 37) % 2) residual |= uint64_t(1) << split;
                    input[i] = static_cast<T>(uint64_t(base) + residual);
                }
                C codec(input, gef::BlockParameters<T>{base, uint8_t(width), uint8_t(split)});
                ASSERT_EQ(codec.b, split);
                ASSERT_EQ(codec.h, unsigned(with_high));
                for (size_t start = 0; start <= 65; ++start) {
                    for (size_t request : {size_t(0), size_t(1), size_t(63), size_t(64),
                                           size_t(65), size_t(129), input.size() + 7}) {
                        const size_t expected = std::min(request, input.size() - start);
                        const T sentinel = static_cast<T>(73);
                        std::vector<T> output(request + 3, sentinel);
                        // Offset output exercises unaligned stores and checks
                        // both sides of exactly the returned output range.
                        ASSERT_EQ(codec.get_elements_into(start, request, output.data() + 1), expected);
                        ASSERT_TRUE(std::equal(input.begin() + start, input.begin() + start + expected,
                                               output.begin() + 1));
                        ASSERT_EQ(output[0], sentinel);
                        ASSERT_TRUE(std::all_of(output.begin() + expected + 1, output.end(),
                                                [=](T value) { return value == sentinel; }));
                    }
                }
                std::vector<T> output(input.size() + 1, T(73));
                ASSERT_EQ(codec.get_elements(0, input.size(), output), input.size());
                ASSERT_TRUE(std::equal(input.begin(), input.end(), output.begin()));
                ASSERT_EQ(output.back(), T(73));
                ASSERT_EQ(codec.get_elements_into(input.size(), 9, nullptr), 0);
            }
        }
    }
}

template<typename T>
void check_exact_low_blocks() {
    using C = gef::internal::RLE_GEF<T>;
    std::vector<unsigned> splits{8, 16};
    for (unsigned split = 17; split <= 32; ++split) splits.push_back(split);
    for (unsigned split : splits) {
        const uint64_t low_mask = (uint64_t(1) << split) - 1;
        for (size_t count : {size_t(64), size_t(128), size_t(129)}) {
            for (bool with_high : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "split=" << split << " count=" << count
                                                 << " high=" << with_high);
                const T base = std::numeric_limits<T>::min();
                std::vector<T> input(count);
                for (size_t i = 0; i < count; ++i) {
                    uint64_t residual = (i * 0x9e3779b97f4a7c15ULL) & low_mask;
                    if (with_high && (i / 7) % 2) residual |= uint64_t(1) << split;
                    input[i] = static_cast<T>(uint64_t(base) + residual);
                }
                C codec(input, gef::BlockParameters<T>{base, uint8_t(split + with_high), uint8_t(split)});
                std::vector<T> output(count + 2, T(73));
                ASSERT_EQ(codec.get_elements_into(0, count, output.data() + 1), count);
                EXPECT_TRUE(std::equal(input.begin(), input.end(), output.begin() + 1));
                EXPECT_EQ(output.front(), T(73));
                EXPECT_EQ(output.back(), T(73));
            }
        }
    }
}
}

TEST(RLEPackedWindow, Signed64) { check_packed_ranges<int64_t>(); }
TEST(RLEPackedWindow, Unsigned64) { check_packed_ranges<uint64_t>(); }
TEST(RLEPackedWindow, Signed32Fallback) { check_packed_ranges<int32_t>(); }
TEST(RLEPackedWindow, Unsigned16Fallback) { check_packed_ranges<uint16_t>(); }
TEST(RLEPackedWindow, ExactBlocksSigned64) { check_exact_low_blocks<int64_t>(); }
TEST(RLEPackedWindow, ExactBlocksUnsigned64) { check_exact_low_blocks<uint64_t>(); }
