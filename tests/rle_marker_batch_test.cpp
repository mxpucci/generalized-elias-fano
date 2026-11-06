#include <gtest/gtest.h>
#include "gef/RLE_GEF.hpp"
#include <array>
#include <filesystem>
#include <limits>
#include <type_traits>
#include <vector>

namespace {
uint64_t width_mask(unsigned width) {
    return width == 64 ? ~uint64_t(0) : (uint64_t(1) << width) - 1;
}

template<typename T>
std::vector<T> marker_input(unsigned split, uint64_t tail) {
    constexpr unsigned width = sizeof(T) * 8;
    constexpr T base = std::numeric_limits<T>::min();
    const uint64_t low_mask = width_mask(split);
    const uint64_t high_mask = width_mask(width - split);
    const std::array<uint64_t, 5> markers{
        0x8102040810204083ULL, 0, ~uint64_t(0), 0x0080018001008100ULL, tail};
    std::vector<T> input(4 * 64 + 13);
    uint64_t high = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        if ((markers[i / 64] >> (i & 63)) & 1) high ^= high_mask;
        uint64_t residual = (split == 64 ? 0 : high << split) |
                            ((i * 0x9e3779b97f4a7c15ULL) & low_mask);
        // Include the entire signed/unsigned domain at every split.
        if (i == 0) residual = 0;
        if (i == 1) residual = width_mask(width);
        input[i] = static_cast<T>(static_cast<uint64_t>(base) + residual);
    }
    return input;
}

template<typename T, typename C>
void check_full_and_clipped(const C& codec, const std::vector<T>& input) {
    const T sentinel = static_cast<T>(73);
    const size_t request = input.size() + 9;
    std::vector<T> output(request + 1, sentinel);
    ASSERT_EQ(codec.get_elements_into(0, request, output.data()), input.size());
    EXPECT_TRUE(std::equal(input.begin(), input.end(), output.begin()));
    EXPECT_TRUE(std::all_of(output.begin() + input.size(), output.end(),
                           [=](T value) { return value == sentinel; }));
    std::fill(output.begin(), output.end(), sentinel);
    ASSERT_EQ(codec.get_elements(0, request, output), input.size());
    EXPECT_TRUE(std::equal(input.begin(), input.end(), output.begin()));
    EXPECT_EQ(output[input.size()], sentinel);
    // The support-free version decodes a prefix before copying this range.
    std::fill(output.begin(), output.end(), sentinel);
    ASSERT_EQ(codec.get_elements_into(7, request, output.data()), input.size() - 7);
    EXPECT_TRUE(std::equal(input.begin() + 7, input.end(), output.begin()));
    EXPECT_EQ(output[input.size() - 7], sentinel);
}

template<typename T, typename BitVector = PastaRankBitVector, bool RandomAccess = true>
void check_types_and_tails() {
    constexpr unsigned width = sizeof(T) * 8;
    using C = gef::internal::RLE_GEF<T, BitVector, RandomAccess>;
    for (unsigned split : {0u, 1u, width - 1, width}) {
        for (uint64_t tail : {uint64_t(0), uint64_t(1), uint64_t(1) << 12, uint64_t(0x1555)}) {
            SCOPED_TRACE(::testing::Message() << "split=" << split << " tail=" << tail);
            const auto input = marker_input<T>(split, tail);
            C codec(input, gef::BlockParameters<T>{std::numeric_limits<T>::min(),
                                                   uint8_t(width), uint8_t(split)});
            check_full_and_clipped(codec, input);
        }
    }
}

template<typename BitVector, bool RandomAccess>
void check_packed_roundtrip(const char* name) {
    using C = gef::internal::RLE_GEF<int64_t, BitVector, RandomAccess>;
    const auto input = marker_input<int64_t>(5, 0x1001);
    C codec(input, gef::BlockParameters<int64_t>{std::numeric_limits<int64_t>::min(), 64, 5});
    const auto path = std::filesystem::temp_directory_path() / name;
    codec.serialize(path);
    C loaded;
    loaded.load(path);
    std::filesystem::remove(path);
    ASSERT_GT(loaded.h, 0);
    ASSERT_EQ(loaded.b, 5);
    EXPECT_EQ(codec.size_in_bytes(), loaded.size_in_bytes());
    check_full_and_clipped(loaded, input);
}

template<typename T>
void check_every_high_width() {
    using C = gef::internal::RLE_GEF<T>;
    const T base = std::numeric_limits<T>::min();
    for (unsigned high_width = 1; high_width <= 64; ++high_width) {
        SCOPED_TRACE(::testing::Message() << "high_width=" << high_width);
        const unsigned split = 64 - high_width;
        const uint64_t high_mask = width_mask(high_width);
        const uint64_t low_mask = width_mask(split);
        uint64_t rng = 0xdca184259fe037b1ULL;
        auto random = [&] {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            return rng;
        };
        uint64_t high = 0;
        std::vector<T> input(129);
        for (size_t i = 0; i < input.size(); ++i) {
            if (i != 0 && i % 2 == 0) {
                uint64_t next = i == 2 ? high_mask : random() & high_mask;
                if (next == high) next ^= 1;
                high = next;
            }
            const uint64_t residual = (high << split) | (random() & low_mask);
            input[i] = static_cast<T>(static_cast<uint64_t>(base) + residual);
        }
        // 65 high fields leave a partial final packed word for every h < 64.
        C codec(input, gef::BlockParameters<T>{base, 64, uint8_t(split)});
        ASSERT_EQ(codec.H.size(), 65);
        ASSERT_EQ(codec.H.width(), high_width);
        check_full_and_clipped(codec, input);
    }
}
}

template<typename T> class RLEMarkerTypes : public ::testing::Test {};
using MarkerTypes = ::testing::Types<int8_t, uint8_t, int16_t, uint16_t,
                                    int32_t, uint32_t, int64_t, uint64_t>;
TYPED_TEST_CASE(RLEMarkerTypes, MarkerTypes);
TYPED_TEST(RLEMarkerTypes, FullWidthExtremesAndTailMasks) { check_types_and_tails<TypeParam>(); }

TEST(RLEMarkerBackends, SDSL) { check_types_and_tails<int64_t, SDSLBitVector>(); }
TEST(RLEMarkerBackends, SupportFreePasta) { check_types_and_tails<int64_t, PastaRankBitVector, false>(); }
TEST(RLEMarkerBackends, SupportFreeSDSL) { check_types_and_tails<int64_t, SDSLBitVector, false>(); }
TEST(RLEMarkerBackends, PackedPastaRoundtrip) {
    check_packed_roundtrip<PastaRankBitVector, true>("gef-marker-pasta.bin");
}
TEST(RLEMarkerBackends, PackedSDSLRoundtrip) {
    check_packed_roundtrip<SDSLBitVector, true>("gef-marker-sdsl.bin");
}
TEST(RLEMarkerBackends, PackedSupportFreeRoundtrip) {
    check_packed_roundtrip<PastaRankBitVector, false>("gef-marker-support-free.bin");
}

TEST(RLEMarkerPackedHigh, EveryWidthSigned64) { check_every_high_width<int64_t>(); }
TEST(RLEMarkerPackedHigh, EveryWidthUnsigned64) { check_every_high_width<uint64_t>(); }
