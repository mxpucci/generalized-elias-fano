#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>
#include "gef/RLE_GEF.hpp"
#include "gef/gef.hpp"

namespace {
// Values whose high part (above `split`) starts a new run with probability
// `run_probability`, over the full 64-bit range so that every split 0..64
// can be forced; low parts are random.
std::vector<uint64_t> runs_of(size_t n, unsigned split, double run_probability, std::mt19937_64& rng) {
    std::bernoulli_distribution new_run(run_probability);
    const uint64_t low_mask = split == 64 ? ~uint64_t(0) : (uint64_t(1) << split) - 1;
    std::vector<uint64_t> values(n);
    uint64_t high = rng() & ~low_mask;
    for (auto& value : values) {
        if (new_run(rng)) high = rng() & ~low_mask;
        value = high | (rng() & low_mask);
    }
    return values;
}

template<class Codec, class T>
void check_ranges(const Codec& codec, const std::vector<T>& values) {
    const size_t n = values.size();
    std::vector<T> decoded(n + 1);
    ASSERT_EQ(codec.get_elements(0, n, decoded), n);
    ASSERT_TRUE(std::equal(values.begin(), values.end(), decoded.begin())) << "whole vector";
    const size_t starts[] = {0, 1, 2, 63, 64, 65, 127, 128, 129, 200, n / 2, n - 300, n - 130, n - 129,
                             n - 128, n - 65, n - 64, n - 63, n - 2, n - 1};
    const size_t lengths[] = {1, 2, 7, 63, 64, 65, 127, 128, 129, 191, 192, 300, 1000, n};
    for (size_t start : starts) {
        if (start >= n) continue;
        for (size_t length : lengths) {
            const size_t count = std::min(length, n - start);
            std::fill(decoded.begin(), decoded.end(), T(7));
            ASSERT_EQ(codec.get_elements(start, length, decoded), count);
            for (size_t i = 0; i < count; ++i)
                ASSERT_EQ(decoded[i], values[start + i]) << "start=" << start << " length=" << length << " i=" << i;
            ASSERT_EQ(decoded[count], T(7)) << "wrote past the range";
        }
    }
}
} // namespace

// Every low/high split, at run densities from very sparse to a run per value
// (all-marker bytes), with range starts and lengths around block edges, the
// first and last blocks, single values and the whole vector.
TEST(RleBlockDecode, EverySplitAndDensity) {
    std::mt19937_64 rng(31415);
    for (unsigned split = 0; split <= 64; ++split) {
        for (double density : {0.02, 0.2, 0.6, 1.0}) {
            SCOPED_TRACE(::testing::Message() << "split=" << split << " density=" << density);
            // Enough runs that the block kernel (which leaves the last 72 runs
            // to the run-list path) decodes most blocks at every density.
            const size_t n = (density < 0.1 ? 8000 : 1500) + rng() % 600;
            const auto values = runs_of(n, split, density, rng);
            const gef::BlockParameters<uint64_t> parameters{0, 64, static_cast<uint8_t>(split)};
            check_ranges(gef::internal::RLE_GEF<uint64_t>(values, parameters), values);
        }
    }
}

TEST(RleBlockDecode, OptimalSplitsSignedAndNarrowTypes) {
    std::mt19937_64 rng(2718);
    for (double density : {0.05, 0.3}) {
        std::vector<int64_t> wide(5000);
        int64_t value = -1000000;
        for (auto& x : wide) {
            if (std::bernoulli_distribution(density)(rng)) value += static_cast<int64_t>(rng() % 200001) - 100000;
            x = value + static_cast<int64_t>(rng() % 64);
        }
        check_ranges(gef::internal::RLE_GEF<int64_t>(wide), wide);
        std::vector<int32_t> narrow(5000);
        for (size_t i = 0; i < narrow.size(); ++i) narrow[i] = static_cast<int32_t>(wide[i] / 3);
        check_ranges(gef::internal::RLE_GEF<int32_t>(narrow), narrow);
    }
}

TEST(RleBlockDecode, LargeAndPartitionedVectors) {
    // Above 65535 values the RLE keeps Pasta rank on B; uniform partitions
    // split ranges across codecs.
    std::mt19937_64 rng(1618);
    for (unsigned split : {0u, 7u, 21u, 44u}) {
        const auto values = runs_of(70001, split, 0.25, rng);
        const gef::BlockParameters<uint64_t> parameters{0, 64, static_cast<uint8_t>(split)};
        check_ranges(gef::internal::RLE_GEF<uint64_t>(values, parameters), values);
    }
    std::vector<int64_t> series(100000);
    int64_t value = 0;
    for (auto& x : series) {
        if (rng() % 5 == 0) value += static_cast<int64_t>(rng() % 4001) - 2000;
        x = value * 16 + static_cast<int64_t>(rng() % 16);
    }
    check_ranges(gef::RLE_GEF<int64_t>(series), series);
}
