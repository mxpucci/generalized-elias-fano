#include <gtest/gtest.h>
#include "gef/RLE_GEF.hpp"
#include <array>
#include <bit>
#include <filesystem>
#include <limits>
#include <type_traits>
#include <utility>
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
    const std::array<std::pair<size_t, size_t>, 12> cases{{
        {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {7, 1},
        {8, 1}, {9, 1}, {63, 1}, {64, 1}, {65, 1}, {65, 2}}};
    for (unsigned high_width = 1; high_width <= 64; ++high_width) {
        SCOPED_TRACE(::testing::Message() << "high_width=" << high_width);
        const unsigned split = 64 - high_width;
        const uint64_t high_mask = width_mask(high_width);
        const uint64_t low_mask = width_mask(split);
        for (const auto& [high_count, stride] : cases) {
            SCOPED_TRACE(::testing::Message() << "high_count=" << high_count << " stride=" << stride);
            uint64_t rng = 0xdca184259fe037b1ULL;
            auto random = [&] {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                return rng;
            };
            uint64_t high = 0;
            std::vector<T> input(129);
            for (size_t i = 0; i < input.size(); ++i) {
                if (i != 0 && i % stride == 0 && i / stride < high_count) {
                    uint64_t next = i == stride ? high_mask : random() & high_mask;
                    if (next == high) next ^= 1;
                    high = next;
                }
                const uint64_t residual = (high << split) | (random() & low_mask);
                input[i] = static_cast<T>(static_cast<uint64_t>(base) + residual);
            }
            // Stride 1 exhausts H before remaining zero-marker words and tail;
            // stride 2 puts the final H field in the tail. 64 fields end on an
            // exact word; 65 leave a partial word for h < 64.
            C codec(input, gef::BlockParameters<T>{base, 64, uint8_t(split)});
            ASSERT_EQ(codec.H.size(), high_count);
            ASSERT_EQ(codec.H.width(), high_width);
            check_full_and_clipped(codec, input);
        }
    }
}

template<typename T>
void check_high_batch_offsets() {
    using C = gef::internal::RLE_GEF<T>;
    const T base = std::numeric_limits<T>::min();
    for (unsigned high_width = 1; high_width <= 16; ++high_width) {
        const unsigned split = 64 - high_width;
        const uint64_t high_mask = width_mask(high_width);
        const uint64_t low_mask = width_mask(split);
        for (unsigned prefix_runs : {1u, 2u, 3u}) {
            for (uint64_t tail : {uint64_t(0), uint64_t(1)}) {
                SCOPED_TRACE(::testing::Message() << "high_width=" << high_width
                    << " prefix_runs=" << prefix_runs << " tail=" << tail);
                // A short first high group leaves a nonzero bit offset for
                // the next word's dense batches, even at high width 16.
                const std::array<uint64_t, 4> markers{
                    (uint64_t(1) << prefix_runs) - 1, ~uint64_t(0), 0, tail};
                std::vector<T> input(3 * 64 + 1);
                uint64_t high = 0;
                for (size_t i = 0; i < input.size(); ++i) {
                    if (i != 0 && ((markers[i / 64] >> (i & 63)) & 1)) high ^= high_mask;
                    const uint64_t residual = (high << split) |
                        ((i * 0x9e3779b97f4a7c15ULL) & low_mask);
                    input[i] = static_cast<T>(static_cast<uint64_t>(base) + residual);
                }
                C codec(input, gef::BlockParameters<T>{base, 64, uint8_t(split)});
                ASSERT_EQ(codec.H.size(), prefix_runs + 64 + tail);
                ASSERT_EQ(codec.H.width(), high_width);
                std::vector<T> output(input.size() + 2, T(73));
                ASSERT_EQ(codec.get_elements_into(0, input.size(), output.data() + 1), input.size());
                EXPECT_TRUE(std::equal(input.begin(), input.end(), output.begin() + 1));
                EXPECT_EQ(output.front(), T(73));
                EXPECT_EQ(output.back(), T(73));
            }
        }
    }
}

template<typename T>
void check_fused_high_batch_offsets() {
    using C = gef::internal::RLE_GEF<T>;
    for (unsigned split : {1u, 5u, 8u, 13u, 16u}) {
        for (unsigned high_width : {3u, 6u, 10u, 16u}) {
            const uint64_t high_mask = width_mask(high_width);
            const uint64_t low_mask = width_mask(split);
            const uint64_t residual_mask = width_mask(split + high_width);
            const T base = std::is_signed_v<T> ? std::numeric_limits<T>::min()
                : static_cast<T>(std::numeric_limits<T>::max() - residual_mask);
            for (unsigned prefix_runs : {1u, 2u, 3u}) {
                for (uint64_t tail : {uint64_t(0), uint64_t(1), uint64_t(1) << 12, uint64_t(0x1555)}) {
                    SCOPED_TRACE(::testing::Message() << "split=" << split << " high_width=" << high_width
                        << " prefix_runs=" << prefix_runs << " tail=" << tail);
                    // Small low splits select the fused C4 path. Prefix H counts
                    // leave the next dense word at distinct packed-H offsets;
                    // the zero word and scalar tail check carry/cursor handoff.
                    const std::array<uint64_t, 4> markers{
                        (uint64_t(1) << prefix_runs) - 1, ~uint64_t(0), 0, tail};
                    std::vector<T> input(3 * 64 + 13);
                    uint64_t high = 0;
                    for (size_t i = 0; i < input.size(); ++i) {
                        if (i != 0 && ((markers[i / 64] >> (i & 63)) & 1)) high ^= high_mask;
                        const uint64_t residual = (high << split) |
                            ((i * 0x9e3779b97f4a7c15ULL) & low_mask);
                        input[i] = static_cast<T>(static_cast<uint64_t>(base) + residual);
                    }
                    C codec(input, gef::BlockParameters<T>{base, uint8_t(split + high_width), uint8_t(split)});
                    ASSERT_EQ(codec.H.size(), prefix_runs + 64 + std::popcount(tail));
                    ASSERT_EQ(codec.H.width(), high_width);
                    ASSERT_EQ(codec.b, split);
                    for (size_t requested : {size_t(64), size_t(128), size_t(192), input.size(), input.size() + 9}) {
                        const size_t count = std::min(requested, input.size());
                        std::vector<T> output(count + 2, T(73));
                        ASSERT_EQ(codec.get_elements_into(0, requested, output.data() + 1), count);
                        EXPECT_TRUE(std::equal(input.begin(), input.begin() + count, output.begin() + 1));
                        EXPECT_EQ(output.front(), T(73));
                        EXPECT_EQ(output.back(), T(73));
                    }
                    check_full_and_clipped(codec, input);
                }
            }
        }
    }
}

template<typename T>
void check_narrow_marker_bytes() {
    using C = gef::internal::RLE_GEF<T>;
    for (unsigned split = 1; split <= 16; ++split) {
        const uint64_t low_mask = width_mask(split);
        const uint64_t residual_mask = width_mask(split + 1);
        const T base = std::is_signed_v<T> ? std::numeric_limits<T>::min()
            : static_cast<T>(std::numeric_limits<T>::max() - residual_mask);
        for (unsigned byte = 0; byte < 256; ++byte) {
            SCOPED_TRACE(::testing::Message() << "split=" << split << " byte=" << byte);
            const uint64_t repeated = byte * 0x0101010101010101ULL;
            const std::array<uint64_t, 4> markers{
                0x0123456789abcdefULL, repeated, ~repeated, 1};
            std::vector<T> input(3 * 64 + 1);
            uint64_t high = 0;
            for (size_t i = 0; i < input.size(); ++i) {
                if ((markers[i / 64] >> (i & 63)) & 1) high ^= 1;
                const uint64_t residual = (high << split) |
                    ((i * 0x9e3779b97f4a7c15ULL) & low_mask);
                input[i] = static_cast<T>(static_cast<uint64_t>(base) + residual);
            }
            C codec(input, gef::BlockParameters<T>{base, uint8_t(split + 1), uint8_t(split)});
            std::vector<T> output(input.size() + 2, T(73));
            ASSERT_EQ(codec.get_elements_into(0, input.size(), output.data() + 1), input.size());
            EXPECT_TRUE(std::equal(input.begin(), input.end(), output.begin() + 1));
            EXPECT_EQ(output.front(), T(73));
            EXPECT_EQ(output.back(), T(73));
        }
    }
}

template<typename T, typename BitVector = PastaRankBitVector, bool RandomAccess = true>
void check_partial_marker_ranges(bool exhaustive, const char* roundtrip_name, bool zero_split = false) {
    using C = gef::internal::RLE_GEF<T, BitVector, RandomAccess>;
    for (unsigned split : {0u, 1u, 5u, 8u, 13u, 16u}) {
        if ((split == 0) != zero_split) continue;
        for (unsigned high_width : {1u, 3u, 6u, 10u, 16u, 17u, 19u, 31u, 32u, 48u, 63u, 64u}) {
            const bool zero_width = high_width == 17 || high_width == 19 || high_width == 32 || high_width == 64;
            if (zero_width != zero_split) continue;
            if (split + high_width > 64) continue;
            if (!exhaustive && !zero_split && (split != 5 || high_width != 6)) continue;
            const uint64_t residual_mask = width_mask(split + high_width);
            const T base = std::is_signed_v<T> ? std::numeric_limits<T>::min()
                : static_cast<T>(std::numeric_limits<T>::max() - residual_mask);
            // Dense/sparse words surround empty words; the final H field is
            // consumed before two empty words and the partial final word.
            const std::array<uint64_t, 9> markers{
                0x8102040810204083ULL, 0, ~uint64_t(0), 0x8000000000000001ULL,
                0, ~uint64_t(0), 0, 0, 0};
            std::vector<T> input(8 * 64 + 13);
            uint64_t high = 0;
            for (size_t i = 0; i < input.size(); ++i) {
                if (i != 0 && ((markers[i / 64] >> (i & 63)) & 1))
                    high ^= width_mask(high_width);
                const uint64_t residual = (high << split) |
                    ((i * 0x9e3779b97f4a7c15ULL) & width_mask(split));
                input[i] = static_cast<T>(static_cast<uint64_t>(base) + residual);
            }
            C codec(input, gef::BlockParameters<T>{base, uint8_t(split + high_width), uint8_t(split)});
            ASSERT_EQ(codec.b, split);
            ASSERT_EQ(codec.H.width(), high_width);
            ASSERT_GT(codec.h, 0);
            auto check = [&](const C& tested) {
                auto range = [&](size_t start, size_t requested) {
                    SCOPED_TRACE(::testing::Message() << "split=" << split << " high=" << high_width
                        << " start=" << start << " count=" << requested);
                    const size_t actual = start < input.size()
                        ? std::min(requested, input.size() - start) : 0;
                    const T sentinel = T(73);
                    std::vector<T> output(requested + 2, sentinel);
                    ASSERT_EQ(tested.get_elements_into(start, requested, output.data() + 1), actual);
                    EXPECT_TRUE(std::equal(input.begin() + std::min(start, input.size()),
                        input.begin() + std::min(start, input.size()) + actual, output.begin() + 1));
                    EXPECT_EQ(output.front(), sentinel);
                    EXPECT_TRUE(std::all_of(output.begin() + actual + 1, output.end(),
                        [=](T value) { return value == sentinel; }));
                    std::fill(output.begin(), output.end(), sentinel);
                    ASSERT_EQ(tested.get_elements(start, requested, output), actual);
                    EXPECT_TRUE(std::equal(input.begin() + std::min(start, input.size()),
                        input.begin() + std::min(start, input.size()) + actual, output.begin()));
                    EXPECT_TRUE(std::all_of(output.begin() + actual, output.end(),
                        [=](T value) { return value == sentinel; }));
                };
                // Every initial word phase, including starts exactly on a
                // marker, and both sides of the partial-decoder length gate.
                for (size_t start = 0; start < 128; ++start)
                    for (size_t count : {size_t(127), size_t(128), size_t(129), size_t(160), size_t(257)})
                        range(start, count);
                for (size_t start : {size_t(255), size_t(256), size_t(257), size_t(383), size_t(384),
                                     size_t(385), size_t(447), size_t(448), size_t(449), size_t(512), input.size()})
                    for (size_t count : {size_t(0), size_t(1), size_t(80), size_t(128), input.size() + 9})
                        range(start, count);
                range(0, input.size());
            };
            check(codec);
            if (zero_split || (split == 5 && high_width == 6)) {
                const auto path = std::filesystem::temp_directory_path() / roundtrip_name;
                codec.serialize(path);
                C loaded;
                loaded.load(path);
                std::filesystem::remove(path);
                ASSERT_EQ(loaded.b, split);
                ASSERT_EQ(loaded.H.width(), high_width);
                EXPECT_EQ(loaded.size_in_bytes(), codec.size_in_bytes());
                check(loaded);
            }
        }
    }
}

template<typename T, bool RandomAccess = true>
void check_wide_high_lookup() {
    using C = gef::internal::RLE_GEF<T, PastaRankBitVector, RandomAccess>;
    const std::array<std::pair<unsigned, unsigned>, 9> widths{{
        {0, 1}, {0, 64}, {1, 63}, {5, 6}, {13, 1},
        {16, 16}, {17, 19}, {31, 32}, {63, 1}}};
    for (auto [split, high_width] : widths) {
        const uint64_t residual_mask = width_mask(split + high_width);
        const T base = std::is_signed_v<T> ? std::numeric_limits<T>::min()
            : static_cast<T>(std::numeric_limits<T>::max() - residual_mask);
        for (unsigned byte = 0; byte < 256; ++byte) {
            SCOPED_TRACE(::testing::Message() << "split=" << split << " high=" << high_width
                << " byte=" << byte << " random_access=" << RandomAccess);
            // The final byte is preceded by 56 real runs: its wide source
            // load reaches index63, or index64 for the dense255 exception.
            // Subsequent zero words read initialized but unselected slots
            // left from the dense word; no additional H fields exist.
            const uint64_t target = ((UINT64_C(1) << 56) - 1) | (uint64_t(byte) << 56);
            const std::array<uint64_t, 5> markers{~uint64_t(0), target, 0, 0, 0};
            std::vector<T> input(4 * 64 + 13);
            uint64_t high = 0;
            for (size_t i = 0; i < input.size(); ++i) {
                if (i != 0 && ((markers[i / 64] >> (i & 63)) & 1))
                    high ^= width_mask(high_width);
                uint64_t low = (i * UINT64_C(0x9e3779b97f4a7c15)) & width_mask(split);
                if (i == 1) low = width_mask(split);
                input[i] = std::bit_cast<T>(static_cast<uint64_t>(base) + (high << split) + low);
            }
            C codec(input, gef::BlockParameters<T>{base, uint8_t(split + high_width), uint8_t(split)});
            ASSERT_EQ(codec.b, split);
            ASSERT_EQ(codec.H.width(), high_width);
            ASSERT_EQ(codec.H.size(), 120 + std::popcount(byte));
            if (split == 0) ASSERT_EQ(codec.L.size(), 0);
            for (size_t start : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(128)}) {
                for (size_t request : {size_t(127), size_t(128), size_t(129), input.size() + 9}) {
                    const size_t actual = std::min(request, input.size() - start);
                    std::vector<T> output(actual + 2, T(73));
                    ASSERT_EQ(codec.get_elements_into(start, request, output.data() + 1), actual);
                    EXPECT_TRUE(std::equal(input.begin() + start, input.begin() + start + actual,
                                           output.begin() + 1));
                    EXPECT_EQ(output.front(), T(73));
                    EXPECT_EQ(output.back(), T(73));
                }
            }
        }
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

TEST(RLEMarkerBytes, NarrowSigned64) { check_narrow_marker_bytes<int64_t>(); }
TEST(RLEMarkerBytes, NarrowUnsigned64) { check_narrow_marker_bytes<uint64_t>(); }

TEST(RLEMarkerPackedHigh, BatchOffsetsSigned64) { check_high_batch_offsets<int64_t>(); }
TEST(RLEMarkerPackedHigh, BatchOffsetsUnsigned64) { check_high_batch_offsets<uint64_t>(); }
TEST(RLEMarkerPackedHigh, FusedBatchOffsetsSigned64) { check_fused_high_batch_offsets<int64_t>(); }
TEST(RLEMarkerPackedHigh, FusedBatchOffsetsUnsigned64) { check_fused_high_batch_offsets<uint64_t>(); }

TEST(RLEMarkerPartial, Signed64) {
    check_partial_marker_ranges<int64_t>(true, "gef-partial-signed.bin");
}
TEST(RLEMarkerPartial, Unsigned64) {
    check_partial_marker_ranges<uint64_t>(true, "gef-partial-unsigned.bin");
}
TEST(RLEMarkerPartial, SDSL) {
    check_partial_marker_ranges<int64_t, SDSLBitVector>(false, "gef-partial-sdsl.bin");
}
TEST(RLEMarkerPartial, SupportFreePasta) {
    check_partial_marker_ranges<int64_t, PastaRankBitVector, false>(false, "gef-partial-no-ra.bin");
}
TEST(RLEMarkerPartial, SupportFreeSDSL) {
    check_partial_marker_ranges<int64_t, SDSLBitVector, false>(false, "gef-partial-sdsl-no-ra.bin");
}

TEST(RLEMarkerPartialZero, Signed64) {
    check_partial_marker_ranges<int64_t>(true, "gef-partial-zero-signed.bin", true);
}
TEST(RLEMarkerPartialZero, Unsigned64) {
    check_partial_marker_ranges<uint64_t>(true, "gef-partial-zero-unsigned.bin", true);
}
TEST(RLEMarkerPartialZero, SDSL) {
    check_partial_marker_ranges<int64_t, SDSLBitVector>(false, "gef-partial-zero-sdsl.bin", true);
}
TEST(RLEMarkerPartialZero, SupportFreePasta) {
    check_partial_marker_ranges<int64_t, PastaRankBitVector, false>(false, "gef-partial-zero-no-ra.bin", true);
}
TEST(RLEMarkerPartialZero, SupportFreeSDSL) {
    check_partial_marker_ranges<int64_t, SDSLBitVector, false>(false, "gef-partial-zero-sdsl-no-ra.bin", true);
}

TEST(RLEMarkerWideLookup, Signed64) { check_wide_high_lookup<int64_t>(); }
TEST(RLEMarkerWideLookup, Unsigned64) { check_wide_high_lookup<uint64_t>(); }
TEST(RLEMarkerWideLookup, SupportFree) { check_wide_high_lookup<int64_t, false>(); }
