#include <gtest/gtest.h>
#include <gef/gef.hpp>
#include <vector>

template<class Codec>
void check_ranges(const std::vector<int64_t>& data) {
    Codec codec(data);
    for (size_t start : {size_t(0), size_t(1), size_t(31), size_t(32), size_t(63), data.size()-1}) {
        for (size_t count : {size_t(0), size_t(1), size_t(33), size_t(97), data.size()}) {
            std::vector<int64_t> out(count + 1, -987654);
            const size_t expected = std::min(count, data.size() - start);
            ASSERT_EQ(codec.get_elements(start, count, out), expected);
            for (size_t i = 0; i < expected; ++i) ASSERT_EQ(out[i], data[start+i]);
            ASSERT_EQ(out[expected], -987654);
        }
    }
}

TEST(UniformDirectDecode, CrossPartitionRangesAndSupportFreeFallback) {
    std::vector<int64_t> data(327);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = int64_t((i * 773) % 1021) - 500;
    check_ranges<gef::RLE_GEF<int64_t, 32>>(data);
    check_ranges<gef::B_STAR_GEF_APPROXIMATE<int64_t, 32>>(data);
    check_ranges<gef::RLE_GEF<int64_t, 32, false>>(data);
    check_ranges<gef::B_STAR_GEF_APPROXIMATE<int64_t, 32, false>>(data);
}

TEST(UniformDirectDecode, LargeDisjointOutputSlices) {
    std::vector<int64_t> data(250007);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = int64_t((i / 153 + i % 5) % 511) - 300;
    gef::RLE_GEF<int64_t, 32000> rle(data);
    gef::B_STAR_GEF_APPROXIMATE<int64_t, 32000> bstar(data);
    std::vector<int64_t> out(data.size());
    ASSERT_EQ(rle.get_elements(0, data.size(), out), data.size());
    ASSERT_EQ(out, data);
    std::fill(out.begin(), out.end(), -987654);
    ASSERT_EQ(bstar.get_elements(7, data.size(), out), data.size()-7);
    ASSERT_TRUE(std::equal(data.begin()+7, data.end(), out.begin()));
    ASSERT_EQ(out[data.size()-7], -987654);
}
