#include <gtest/gtest.h>
#include "datastructures/PastaBitVector.hpp"
#include <random>
#include <vector>

// Cover word, rank-directory, sampling and partial-final-word boundaries.
TEST(BitVectorNavigation, RankAndSelectAgainstExplicitPositions) {
    std::mt19937_64 rng(7391);
    for (size_t n : {size_t(1), size_t(63), size_t(64), size_t(65),
                     size_t(511), size_t(512), size_t(4095), size_t(4096),
                     size_t(16384), size_t(32769), size_t(150013)}) {
        for (unsigned pattern = 0; pattern < 6; ++pattern) {
            SCOPED_TRACE(n);
            SCOPED_TRACE(pattern);
            std::vector<bool> bits(n);
            std::vector<size_t> ones, zeros;
            for (size_t i = 0; i < n; ++i) {
                bool bit = pattern == 0 ? false : pattern == 1 ? true :
                    pattern == 2 ? (i % 2) : pattern == 3 ? (rng() % 100 == 0) :
                    pattern == 4 ? (rng() % 100 != 0) : ((i / 5001) % 2);
                bits[i] = bit;
                (bit ? ones : zeros).push_back(i);
            }
            PastaGapBitVector gap(bits);
            PastaExceptionBitVector exception(bits);
            PastaRankBitVector ranks(bits);
            gap.enable_select0();
            exception.enable_select1();
            ranks.enable_rank();
            size_t prefix = 0;
            for (size_t i = 0; i <= n; ++i) {
                ASSERT_EQ(ranks.rank(i), prefix);
                if (i < n) prefix += bits[i];
            }
            for (size_t i = 0; i < zeros.size(); ++i)
                ASSERT_EQ(gap.select0(i + 1), zeros[i]);
            for (size_t i = 0; i < ones.size(); ++i)
                ASSERT_EQ(exception.select(i + 1), ones[i]);
            auto copied = gap;
            auto moved = std::move(copied);
            for (size_t i = 0; i < zeros.size(); i += 127)
                ASSERT_EQ(moved.select0(i + 1), zeros[i]);
        }
    }
}

// Also run this test with GEF_SELECT_SAMPLE_RATE=64,512,1024. A partial
// final L1 block can then contain multiple sample intervals without a later
// rank-directory boundary from which to initialize those samples.
TEST(BitVectorNavigation, DenseSamplingInPartialFinalBlock) {
    for (size_t n : {size_t(3000), size_t(4095), size_t(6500), size_t(10001)}) {
        for (bool bit : {false, true}) {
            SCOPED_TRACE(n);
            SCOPED_TRACE(bit);
            const std::vector<bool> bits(n, bit);
            PastaGapBitVector gap(bits);
            PastaExceptionBitVector exception(bits);
            gap.enable_select0();
            exception.enable_select1();
            for (size_t rank = 1; rank <= n; ++rank) {
                if (bit) ASSERT_EQ(exception.select(rank), rank - 1);
                else ASSERT_EQ(gap.select0(rank), rank - 1);
            }
        }
    }
}
