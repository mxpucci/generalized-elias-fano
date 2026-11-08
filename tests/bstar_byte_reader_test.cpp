#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <type_traits>
#include <vector>
#include "gef/gef.hpp"

namespace {
template<class Codec, class T>
void check_ranges(const Codec& codec, const std::vector<T>& values) {
    for (size_t start : {size_t(0), size_t(1), size_t(7), size_t(63), size_t(255), size_t(256)}) {
        if (start >= values.size()) continue;
        for (size_t requested : {size_t(8), size_t(63), size_t(64), size_t(65),
                                 size_t(127), size_t(128), size_t(255), size_t(256),
                                 size_t(257), values.size() + 3}) {
            const size_t count = std::min(requested, values.size() - start);
            std::vector<T> out(requested + 2, T(99));
            ASSERT_EQ(codec.get_elements(start, requested, out), count);
            ASSERT_TRUE(std::equal(values.begin() + start, values.begin() + start + count, out.begin()));
            ASSERT_EQ(out[count], T(99));
        }
    }
}

template<class Bits>
void check_byte_masks() {
    for (unsigned mask = 0; mask < 256; ++mask) {
        SCOPED_TRACE(mask);
        const uint64_t repeated = uint64_t(mask) * UINT64_C(0x0101010101010101);
        // Complemented gap words: arbitrary bytes, a long delimiter-free word,
        // a complete delimiter word, and enough values to cross a256 batch.
        const std::array<uint64_t, 10> zeros{repeated, repeated, 0, repeated,
            UINT64_MAX, repeated, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
        std::vector<uint64_t> highs;
        for (size_t word = 0; word < zeros.size(); ++word)
            for (unsigned bit = 0; bit < 64; ++bit)
                if ((zeros[word] >> bit) & 1)
                    highs.push_back(word * 64 + bit - highs.size());
        for (bool descending : {false, true}) {
            std::vector<uint64_t> values(highs.size());
            for (size_t i = 0; i < values.size(); ++i) {
                const uint64_t high = descending ? highs.back() - highs[i] : highs[i];
                values[i] = (high << 3) | (i & 7);
            }
            gef::internal::B_STAR_GEF<uint64_t, Bits> codec(
                values, gef::BlockParameters<uint64_t>{0, 16, 3});
            check_ranges(codec, values);
        }
    }
}
}

TEST(BStarByteReader, EveryByteMaskPasta) { check_byte_masks<PastaGapBitVector>(); }
TEST(BStarByteReader, EveryByteMaskSdsl) { check_byte_masks<SDSLBitVector>(); }

template<class T> class BStarByteTypes : public ::testing::Test {};
using ByteReaderTypes = ::testing::Types<int8_t, uint8_t, int16_t, uint16_t,
                                       int32_t, uint32_t, int64_t, uint64_t>;
TYPED_TEST_CASE(BStarByteTypes, ByteReaderTypes);
TYPED_TEST(BStarByteTypes, ForcedSplitsAndLongBatches) {
    using T = TypeParam;
    using U = std::make_unsigned_t<T>;
    constexpr unsigned bits = sizeof(T) * 8;
    const T base = std::numeric_limits<T>::lowest();
    for (unsigned split : {0u, 1u, bits - 1, bits}) {
        SCOPED_TRACE(split);
        const U low_mask = split == bits ? std::numeric_limits<U>::max()
                                        : static_cast<U>((uint64_t(1) << split) - 1);
        std::vector<T> values(513);
        for (size_t i = 0; i < values.size(); ++i) {
            // At most one high bit: unary allocations stay small at all widths.
            const U high = split == bits ? U(0) : static_cast<U>((i / 3) & 1);
            const U low = static_cast<U>(i * 73) & low_mask;
            const U residual = static_cast<U>(low | (split == bits ? U(0) : U(uint64_t(high) << split)));
            values[i] = static_cast<T>(static_cast<U>(base) + residual);
        }
        gef::internal::B_STAR_GEF<T> codec(values,
            gef::BlockParameters<T>{base, uint8_t(bits), uint8_t(split)});
        check_ranges(codec, values);
    }
}

TEST(BStarByteReader, PublicOptimalReloadAcrossPartitionBoundary) {
    std::vector<int64_t> values(32000 + 257);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = -1000 + int64_t((i / 3) % 11) * 16 + int64_t(i % 7);
    gef::B_STAR_GEF<int64_t, 32000> codec(values);
    const auto path = std::filesystem::temp_directory_path() / "gef-bstar-byte-reader.bin";
    codec.serialize(path);
    gef::B_STAR_GEF<int64_t, 32000> loaded;
    std::ifstream input(path, std::ios::binary);
    loaded.load(input);
    input.close();
    std::filesystem::remove(path);
    check_ranges(loaded, values);
    std::vector<int64_t> out(520, INT64_MIN);
    ASSERT_EQ(loaded.get_elements(31999, 520, out), values.size() - 31999);
    ASSERT_TRUE(std::equal(values.begin() + 31999, values.end(), out.begin()));
    ASSERT_EQ(out[values.size() - 31999], INT64_MIN);
}

namespace {
template<class T>
void check_packet_pairs() {
    using U = std::make_unsigned_t<T>;
    constexpr std::array<unsigned, 8> bytes{0, 1, 0x80, 0x81, 0x55, 0xaa, 0xfe, 0xff};
    for (unsigned low_byte : bytes) for (unsigned high_byte : bytes) {
        SCOPED_TRACE(::testing::Message() << low_byte << "," << high_byte);
        const uint64_t packet = low_byte | (high_byte << 8);
        const uint64_t word = packet | ((packet ^ 0xffff) << 16) |
            ((packet ^ 0x8001) << 32) | (packet << 48);
        // Independently chosen adjacent bytes, empty/full packets and words,
        // then a partial final physical word. Scalar scanning is the oracle.
        const std::array<uint64_t, 8> zeros{word, 0, UINT64_MAX, ~word,
            word ^ UINT64_C(0x55aa00ffff000081), UINT64_MAX, 0x101, 0x8001};
        std::vector<uint64_t> highs;
        for (size_t w = 0; w < zeros.size(); ++w)
            for (unsigned bit = 0; bit < 64; ++bit)
                if ((zeros[w] >> bit) & 1) highs.push_back(w * 64 + bit - highs.size());
        for (bool descending : {false, true}) for (unsigned split : {0u, 3u, 16u}) {
            const T base = std::numeric_limits<T>::lowest();
            const U mask = (U(1) << split) - 1;
            std::vector<T> values(highs.size());
            for (size_t i = 0; i < values.size(); ++i) {
                const U high = descending ? highs.back() - highs[i] : highs[i];
                values[i] = static_cast<T>(static_cast<U>(base) +
                    ((high << split) | (static_cast<U>(i * 73) & mask)));
            }
            gef::internal::B_STAR_GEF<T> codec(values,
                gef::BlockParameters<T>{base, uint8_t(split + 10), uint8_t(split)});
            check_ranges(codec, values);
            for (size_t start : {size_t(15), size_t(16), size_t(17), size_t(31),
                                 size_t(32), size_t(47), size_t(48), values.size() - 1}) {
                for (size_t request : {size_t(0), size_t(1), size_t(7), size_t(15),
                                       size_t(16), size_t(63), size_t(64), size_t(65), size_t(257)}) {
                    const size_t actual = std::min(request, values.size() - start);
                    std::vector<T> out(actual + 2, T(99));
                    ASSERT_EQ(codec.get_elements_into(start, request, out.data() + 1), actual);
                    ASSERT_TRUE(std::equal(values.begin() + start, values.begin() + start + actual, out.begin() + 1));
                    ASSERT_EQ(out.front(), T(99));
                    ASSERT_EQ(out.back(), T(99));
                }
            }
        }
    }
}
}

TEST(BStarByteReader, IndependentPacketBytesSigned64) { check_packet_pairs<int64_t>(); }
TEST(BStarByteReader, IndependentPacketBytesUnsigned64) { check_packet_pairs<uint64_t>(); }

TEST(BStarByteReader, EveryPhaseAtBatchEndpoints) {
    for (unsigned split : {3u, 7u, 16u}) {
        std::vector<uint64_t> values(8193);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = (((i * 7) % 31) << split) | (i * 73 & ((uint64_t(1) << split) - 1));
        gef::internal::B_STAR_GEF<uint64_t> codec(values,
            gef::BlockParameters<uint64_t>{0, uint8_t(split + 5), uint8_t(split)});
        for (size_t capacity : {size_t(256), size_t(512), size_t(1024), size_t(2048), size_t(4096)}) for (size_t start = 0; start < 64; ++start) {
            const size_t endpoint = capacity - start;
            for (size_t request : {endpoint - 1, endpoint, endpoint + 1}) {
                std::vector<uint64_t> out(request + 2, UINT64_MAX);
                ASSERT_EQ(codec.get_elements_into(start, request, out.data() + 1), request);
                ASSERT_TRUE(std::equal(values.begin() + start, values.begin() + start + request, out.begin() + 1));
                ASSERT_EQ(out.front(), UINT64_MAX);
                ASSERT_EQ(out.back(), UINT64_MAX);
            }
        }
    }
}
