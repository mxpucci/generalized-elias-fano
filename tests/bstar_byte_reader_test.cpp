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
