#include <gtest/gtest.h>

#include "gef/gef.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace {

std::vector<int64_t> make_data(size_t n) {
    std::vector<int64_t> v;
    v.reserve(n);
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int64_t> dist(-1000, 1000);
    int64_t cur = 0;
    for (size_t i = 0; i < n; ++i) {
        // Mix: some monotone-ish drift + noise + duplicates
        cur += (i % 7 == 0) ? 0 : (i % 2 == 0 ? 1 : -1);
        v.push_back(cur + dist(rng));
        if (i % 13 == 0) v.back() = 42; // duplicates
    }
    return v;
}

template <class GEF>
void assert_operator_brackets(const GEF& g, const std::vector<int64_t>& data) {
    ASSERT_EQ(g.size(), data.size());
    const std::vector<size_t> idxs = {0, 1, 5, data.size() / 2, data.size() - 1};
    for (size_t idx : idxs) {
        EXPECT_EQ(g[idx], data[idx]) << "Mismatch at index=" << idx;
    }
}

template <class GEF>
void assert_get_elements(const GEF& g, const std::vector<int64_t>& data) {
    ASSERT_EQ(g.size(), data.size());

    // startIndex = 0
    {
        const size_t start = 0;
        const size_t count = 10;
        std::vector<int64_t> out(count);
        const size_t written = g.get_elements(start, count, out);
        ASSERT_EQ(written, count);
        for (size_t i = 0; i < count; ++i) {
            EXPECT_EQ(out[i], data[start + i]);
        }
    }

    // startIndex > 0 (this should still work even when random_access=false)
    {
        const size_t start = 7;
        const size_t count = 13;
        std::vector<int64_t> out(count);
        const size_t written = g.get_elements(start, count, out);
        ASSERT_EQ(written, count);
        for (size_t i = 0; i < count; ++i) {
            EXPECT_EQ(out[i], data[start + i]);
        }
    }

    // tail clamp
    {
        const size_t start = data.size() - 5;
        const size_t count = 10;
        std::vector<int64_t> out(count);
        const size_t written = g.get_elements(start, count, out);
        ASSERT_EQ(written, 5u);
        for (size_t i = 0; i < written; ++i) {
            EXPECT_EQ(out[i], data[start + i]);
        }
    }
}

} // namespace

// --------------------------------------------------------------------------
// Internal compressors, RandomAccess = false
// --------------------------------------------------------------------------

TEST(RandomAccessFalse_Internal, B_GEF_get_elements_and_brackets) {
    const auto data = make_data(100);
    gef::internal::B_GEF<int64_t, PastaExceptionBitVector, PastaGapBitVector, false> g(data, gef::OPTIMAL_SPLIT_POINT);
    assert_get_elements(g, data);
    assert_operator_brackets(g, data);
}

TEST(RandomAccessFalse_Internal, B_STAR_GEF_get_elements_and_brackets) {
    const auto data = make_data(100);
    gef::internal::B_STAR_GEF<int64_t, PastaGapBitVector, false> g(data, gef::OPTIMAL_SPLIT_POINT);
    assert_get_elements(g, data);
    assert_operator_brackets(g, data);
}

TEST(RandomAccessFalse_Internal, U_GEF_get_elements_and_brackets) {
    const auto data = make_data(100);
    gef::internal::U_GEF<int64_t, PastaExceptionBitVector, PastaGapBitVector, false> g(data, gef::OPTIMAL_SPLIT_POINT);
    assert_get_elements(g, data);
    assert_operator_brackets(g, data);
}

TEST(RandomAccessFalse_Internal, RLE_GEF_get_elements_and_brackets) {
    const auto data = make_data(100);
    gef::internal::RLE_GEF<int64_t, PastaRankBitVector, false> g(data);
    assert_get_elements(g, data);
    assert_operator_brackets(g, data);
}

// --------------------------------------------------------------------------
// Public partitioned wrappers, random_access = false
// (This also exercises UniformPartitioning forwarding + slicing across blocks.)
// --------------------------------------------------------------------------

TEST(RandomAccessFalse_Public, B_GEF_get_elements_and_brackets) {
    const auto data = make_data(100);
    // Small block size to force multiple partitions.
    gef::B_GEF<int64_t, 8, false> g(data, gef::OPTIMAL_SPLIT_POINT);
    assert_get_elements(g, data);
    assert_operator_brackets(g, data);
}

TEST(RandomAccessFalse_Public, B_STAR_GEF_get_elements_and_brackets) {
    const auto data = make_data(100);
    gef::B_STAR_GEF<int64_t, 8, false> g(data, gef::OPTIMAL_SPLIT_POINT);
    assert_get_elements(g, data);
    assert_operator_brackets(g, data);
}

TEST(RandomAccessFalse_Public, U_GEF_get_elements_and_brackets) {
    const auto data = make_data(100);
    gef::U_GEF<int64_t, 8, false> g(data, gef::OPTIMAL_SPLIT_POINT);
    assert_get_elements(g, data);
    assert_operator_brackets(g, data);
}

TEST(RandomAccessFalse_Public, RLE_GEF_get_elements_and_brackets) {
    const auto data = make_data(100);
    gef::RLE_GEF<int64_t, 8, false> g(data);
    assert_get_elements(g, data);
    assert_operator_brackets(g, data);
}


