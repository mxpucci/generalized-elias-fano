#include <gtest/gtest.h>
#include "datastructures/PastaBitVector.hpp"
#include "gef/B_STAR_GEF.hpp"
#include <random>
#include <vector>

namespace {
std::vector<bool> make_bits(size_t n, unsigned pattern, std::mt19937_64& rng) {
    std::vector<bool> bits(n);
    for (size_t i = 0; i < n; ++i) {
        bits[i] = pattern == 0 ? false : pattern == 1 ? true :
                  pattern == 2 ? (i % 2) : pattern == 3 ? (rng() % 100 == 0) :
                  pattern == 4 ? (rng() % 100 != 0) : ((i / 5001) % 2);
    }
    return bits;
}
std::vector<size_t> zeros(const std::vector<bool>& bits) {
    std::vector<size_t> positions;
    for (size_t i = 0; i < bits.size(); ++i) if (!bits[i]) positions.push_back(i);
    return positions;
}
template<class BitVector>
void check_pairs() {
    std::mt19937_64 rng(41971);
    for (size_t n : {size_t(1), size_t(63), size_t(64), size_t(65), size_t(511), size_t(512),
                     size_t(4095), size_t(4096), size_t(6500), size_t(32769), size_t(150013)}) {
        for (unsigned a = 0; a < 6; ++a) for (unsigned b = 0; b < 6; ++b) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " patterns=" << a << ',' << b);
            const auto first_bits = make_bits(n, a, rng);
            const auto second_bits = make_bits(n + 31, b, rng);
            const auto first_zeros = zeros(first_bits), second_zeros = zeros(second_bits);
            BitVector first(first_bits), second(second_bits);
            first.enable_select0(); second.enable_select0();
            for (size_t rank = 1; rank <= std::min(first_zeros.size(), second_zeros.size()); ++rank) {
                const auto pair = first.select0_pair_unchecked(second, rank);
                ASSERT_EQ(pair.first, first_zeros[rank - 1]);
                ASSERT_EQ(pair.second, second_zeros[rank - 1]);
                ASSERT_EQ(pair.first, first.select0_unchecked(rank));
                ASSERT_EQ(pair.second, second.select0_unchecked(rank));
            }
            for (size_t rank = 1; rank <= first_zeros.size(); rank += 127) {
                const auto self = first.select0_pair_unchecked(first, rank);
                ASSERT_EQ(self.first, first_zeros[rank - 1]);
                ASSERT_EQ(self.second, first_zeros[rank - 1]);
            }
        }
    }
}
}

TEST(PairedSelect, ZeroOptimizedAgainstExplicitPositions) { check_pairs<PastaGapBitVector>(); }
TEST(PairedSelect, OneOptimizedAgainstExplicitPositions) { check_pairs<PastaBitVectorT<true, true, true>>(); }

TEST(PairedSelect, BStarGenericFallbackAndPastaAgree) {
    std::mt19937_64 rng(9511);
    for (size_t n : {size_t(1), size_t(65), size_t(4095), size_t(32769)}) {
        std::vector<int64_t> values(n);
        int64_t value = -1700;
        for (auto& x : values) { value += int64_t(rng() % 2001) - 1000; x = value; }
        gef::internal::B_STAR_GEF<int64_t, PastaGapBitVector> paired(values);
        gef::internal::B_STAR_GEF<int64_t, SDSLBitVector> fallback(values);
        for (size_t i = 0; i < n; ++i) {
            ASSERT_EQ(paired[i], values[i]);
            ASSERT_EQ(fallback[i], values[i]);
        }
        std::vector<int64_t> decoded(n);
        EXPECT_EQ(paired.get_elements(0, n, decoded), n);
        EXPECT_EQ(decoded, values);
        EXPECT_EQ(fallback.get_elements(0, n, decoded), n);
        EXPECT_EQ(decoded, values);
    }
}
