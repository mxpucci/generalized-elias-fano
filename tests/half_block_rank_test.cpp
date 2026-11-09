#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include "datastructures/HalfBlockRank.hpp"
#include "gef/RLE_GEF.hpp"
#include "gef/gef.hpp"

using gef::internal::HalfBlockRank;

namespace {
// Exactly the words holding `bits` bits; reads past them are caught by ASan.
std::vector<uint64_t> random_words(size_t bits, double density, std::mt19937_64& rng) {
    std::vector<uint64_t> words((bits + 63) / 64, 0);
    std::bernoulli_distribution one(density);
    for (size_t i = 0; i < bits; ++i)
        if (one(rng)) words[i / 64] |= uint64_t(1) << (i % 64);
    return words;
}

void check_rank(const std::vector<uint64_t>& words, size_t bits) {
    ASSERT_TRUE(HalfBlockRank::supports(bits));
    HalfBlockRank rank;
    EXPECT_FALSE(rank.enabled());
    rank.build(words.data(), bits);
    ASSERT_TRUE(rank.enabled());
    size_t ones = 0;
    for (size_t i = 0; i <= bits; ++i) {
        ASSERT_EQ(rank.rank1(i), ones) << "bits=" << bits << " index=" << i;
        if (i < bits) ones += (words[i / 64] >> (i % 64)) & 1;
    }
    HalfBlockRank moved(std::move(rank));
    EXPECT_FALSE(rank.enabled());
    ASSERT_EQ(moved.rank1(bits), ones);
    HalfBlockRank assigned;
    assigned = std::move(moved);
    EXPECT_FALSE(moved.enabled());
    ASSERT_EQ(assigned.rank1(bits / 2), [&] {
        size_t c = 0;
        for (size_t i = 0; i < bits / 2; ++i) c += (words[i / 64] >> (i % 64)) & 1;
        return c;
    }());
    assigned.clear();
    EXPECT_FALSE(assigned.enabled());
    EXPECT_EQ(assigned.size_in_bytes(), 0u);
}
} // namespace

TEST(HalfBlockRank, BoundarySizesAllDensities) {
    std::mt19937_64 rng(7);
    for (size_t bits : {size_t(1), size_t(63), size_t(64), size_t(65), size_t(255), size_t(256), size_t(257),
                        size_t(511), size_t(512), size_t(513), size_t(767), size_t(768), size_t(769),
                        size_t(4096), size_t(32000), size_t(65535)})
        for (double density : {0.0, 0.01, 0.5, 0.99, 1.0}) {
            SCOPED_TRACE(::testing::Message() << "bits=" << bits << " density=" << density);
            check_rank(random_words(bits, density, rng), bits);
        }
}

TEST(HalfBlockRank, RandomSizes) {
    std::mt19937_64 rng(31);
    for (int round = 0; round < 60; ++round) {
        const size_t bits = 1 + rng() % HalfBlockRank::MAX_BITS;
        check_rank(random_words(bits, std::uniform_real_distribution<double>(0, 1)(rng), rng), bits);
    }
}

TEST(HalfBlockRank, SupportsLimit) {
    EXPECT_TRUE(HalfBlockRank::supports(HalfBlockRank::MAX_BITS));
    EXPECT_FALSE(HalfBlockRank::supports(HalfBlockRank::MAX_BITS + 1));
}

namespace {
template<class Codec, class T>
void expect_decodes(const Codec& codec, const std::vector<T>& values) {
    ASSERT_EQ(codec.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) ASSERT_EQ(codec[i], values[i]) << "index " << i;
    std::vector<T> decoded(values.size());
    ASSERT_EQ(codec.get_elements(0, values.size(), decoded), values.size());
    ASSERT_EQ(decoded, values);
    if (values.size() > 2) {
        const size_t start = values.size() / 3, count = values.size() - start - 1;
        std::vector<T> part(count);
        ASSERT_EQ(codec.get_elements(start, count, part), count);
        ASSERT_TRUE(std::equal(part.begin(), part.end(), values.begin() + start));
    }
}

std::filesystem::path temp_path(const char* name) {
    return std::filesystem::temp_directory_path() / (std::string(name) + std::to_string(::getpid()) + ".bin");
}

template<class Codec, class T>
void check_codec_lifecycle(const Codec& codec, const std::vector<T>& values) {
    expect_decodes(codec, values);
    Codec copy(codec);
    expect_decodes(copy, values);
    Codec assigned;
    assigned = copy;
    expect_decodes(assigned, values);
    Codec moved(std::move(copy));
    expect_decodes(moved, values);
    Codec move_assigned;
    move_assigned = std::move(moved);
    expect_decodes(move_assigned, values);
    const std::vector<T> small{T(1), T(1), T(5)};
    Codec other(small);
    swap(other, move_assigned);
    expect_decodes(other, values);
    expect_decodes(move_assigned, small);
    const auto path = temp_path("half_block_rank_test_");
    codec.serialize(path);
    Codec loaded;
    loaded.load(path);
    std::filesystem::remove(path);
    expect_decodes(loaded, values);
    EXPECT_EQ(loaded.size_in_bytes(), codec.size_in_bytes());
}

// Runs of equal high parts, as RLE expects, around a random walk.
template<class T>
std::vector<T> runs(size_t n, int64_t start, int64_t step, size_t max_run, std::mt19937_64& rng) {
    std::vector<T> values(n);
    int64_t value = start;
    for (size_t i = 0; i < n;) {
        value += static_cast<int64_t>(rng() % (2 * step + 1)) - step;
        for (size_t r = 1 + rng() % max_run; r && i < n; --r, ++i)
            values[i] = static_cast<T>(value + static_cast<int64_t>(rng() % 4));
    }
    return values;
}

// Sizes above HalfBlockRank::MAX_BITS keep Pasta rank on B.
constexpr size_t CODEC_SIZES[] = {1, 2, 64, 257, 513, 4097, 32000, 65535, 65536, 70001};
} // namespace

TEST(HalfBlockRankRle, Int64Int32Uint64) {
    std::mt19937_64 rng(4242);
    for (size_t n : CODEC_SIZES) {
        SCOPED_TRACE(n);
        const auto signed64 = runs<int64_t>(n, -100000, 5000, 12, rng);
        check_codec_lifecycle(gef::internal::RLE_GEF<int64_t>(signed64), signed64);
        const auto signed32 = runs<int32_t>(n, -1000, 300, 3, rng);
        check_codec_lifecycle(gef::internal::RLE_GEF<int32_t>(signed32), signed32);
        std::vector<uint64_t> large(n);
        for (auto& x : large) x = UINT64_MAX - (rng() % (uint64_t(1) << 50));
        check_codec_lifecycle(gef::internal::RLE_GEF<uint64_t>(large), large);
    }
}

TEST(HalfBlockRankRle, ForcedSplitsIncludingNoLowAndNoHighPart) {
    std::mt19937_64 rng(8);
    for (size_t n : {size_t(1), size_t(700), size_t(40000), size_t(66000)}) {
        const auto values = runs<int64_t>(n, 0, 3, 20, rng);
        const int64_t base = *std::min_element(values.begin(), values.end());
        const int64_t top = *std::max_element(values.begin(), values.end());
        const uint8_t width = static_cast<uint8_t>(std::bit_width(static_cast<uint64_t>(top - base)));
        for (uint8_t split : {uint8_t(0), std::min<uint8_t>(1, width), uint8_t(width / 2), width}) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " split=" << int(split));
            // split 0: no low part (b == 0); split == width: no high part (h == 0).
            const gef::BlockParameters<int64_t> parameters{base, width, split};
            check_codec_lifecycle(gef::internal::RLE_GEF<int64_t>(values, parameters), values);
        }
    }
    const std::vector<int64_t> constant(5000, -77);
    check_codec_lifecycle(gef::internal::RLE_GEF<int64_t>(constant), constant);
}

TEST(HalfBlockRankRle, WideFieldsUseTheTwoWordRead) {
    // Low and high parts wider than 57 bits take the two-word field read.
    std::mt19937_64 rng(99);
    std::vector<uint64_t> values(3000);
    for (size_t i = 0; i < values.size(); ++i) values[i] = (i / 7 % 2 ? uint64_t(1) << 63 : 0) | (rng() >> 2);
    check_codec_lifecycle(gef::internal::RLE_GEF<uint64_t>(values), values);
    const gef::BlockParameters<uint64_t> high_only{0, 64, 0};
    check_codec_lifecycle(gef::internal::RLE_GEF<uint64_t>(values, high_only), values);
}

TEST(HalfBlockRankRle, PartitionedCodecs) {
    std::mt19937_64 rng(2718);
    const auto values = runs<int64_t>(250000, 0, 5000, 8, rng);
    expect_decodes(gef::RLE_GEF<int64_t>(values), values);
    expect_decodes(gef::RLE_GEF<int64_t, 100000>(values), values);
    expect_decodes(gef::PARTITIONED_RLE_GEF<int64_t>(values), values);
}

TEST(HalfBlockRankRle, LoadsStreamsWithSerializedPastaSupport) {
    // Earlier versions enabled Pasta rank on B, so their streams set its
    // support flag. Loading one must decode identically without keeping (or
    // counting) the unused support.
    std::mt19937_64 rng(606);
    const auto values = runs<int64_t>(20000, 0, 40, 10, rng);
    const int64_t base = *std::min_element(values.begin(), values.end());
    const int64_t top = *std::max_element(values.begin(), values.end());
    const uint8_t width = static_cast<uint8_t>(std::bit_width(static_cast<uint64_t>(top - base)));
    // b == 0 keeps L out of the stream: h, b, size, base, H (sdsl), then B.
    gef::internal::RLE_GEF<int64_t> codec(values, gef::BlockParameters<int64_t>{base, width, 0});
    const auto path = temp_path("half_block_rank_flag_");
    codec.serialize(path);
    const size_t h_bytes = sdsl::size_in_bytes(codec.H);
    const size_t flag_offset = 2 + sizeof(size_t) + sizeof(int64_t) + h_bytes + sizeof(size_t);
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekg(flag_offset);
        char flag = 1;
        file.read(&flag, 1);
        ASSERT_EQ(flag, 0);
        flag = 1;
        file.seekp(flag_offset);
        file.write(&flag, 1);
    }
    gef::internal::RLE_GEF<int64_t> loaded;
    loaded.load(path);
    std::filesystem::remove(path);
    expect_decodes(loaded, values);
    EXPECT_EQ(loaded.size_in_bytes(), codec.size_in_bytes());
}

namespace {
// The last field of L (index n - 1) and of H (the last run) for every width:
// the one-load read takes eight bytes from a field's first byte, which may
// reach into sdsl's padding word but never past the allocation (checked by
// ASan builds).
void check_last_fields(const gef::internal::RLE_GEF<uint64_t>& codec, const std::vector<uint64_t>& values) {
    const size_t n = values.size();
    ASSERT_EQ(codec[n - 1], values[n - 1]);
    ASSERT_EQ(codec[0], values[0]);
    ASSERT_EQ(codec[n / 2], values[n / 2]);
}
} // namespace

TEST(HalfBlockRankRle, LastFieldsForEveryWidth) {
    std::mt19937_64 rng(5150);
    for (unsigned split = 0; split <= 64; ++split) {
        for (size_t n : {size_t(1), size_t(2), size_t(7), size_t(64), size_t(65), size_t(1 + rng() % 5000)}) {
            SCOPED_TRACE(::testing::Message() << "split=" << split << " n=" << n);
            // Runs of equal high parts with random low parts over all 64 bits.
            std::vector<uint64_t> values(n);
            uint64_t high = rng();
            for (size_t i = 0; i < n; ++i) {
                if (rng() % 3 == 0) high = rng();
                const uint64_t low_mask = split == 64 ? ~uint64_t(0) : (uint64_t(1) << split) - 1;
                values[i] = (high & ~low_mask) | (rng() & low_mask);
            }
            const gef::BlockParameters<uint64_t> parameters{0, 64, static_cast<uint8_t>(split)};
            gef::internal::RLE_GEF<uint64_t> codec(values, parameters);
            check_last_fields(codec, values);
            gef::internal::RLE_GEF<uint64_t> copy(codec);
            check_last_fields(copy, values);
            gef::internal::RLE_GEF<uint64_t> moved(std::move(copy));
            check_last_fields(moved, values);
            gef::internal::RLE_GEF<uint64_t> assigned;
            assigned = moved;
            check_last_fields(assigned, values);
            const auto path = temp_path("half_block_rank_last_");
            codec.serialize(path);
            gef::internal::RLE_GEF<uint64_t> loaded;
            loaded.load(path);
            std::filesystem::remove(path);
            check_last_fields(loaded, values);
        }
    }
}

TEST(HalfBlockRankRle, QueryFieldsShareTheFirstCacheLine) {
    // operator[] reads only these members; they must stay within the first
    // 64 bytes of the 64-byte aligned object.
    gef::internal::RLE_GEF<int64_t> codec;
    const auto offset = [&](const void* member) {
        return static_cast<size_t>(reinterpret_cast<const char*>(member) - reinterpret_cast<const char*>(&codec));
    };
    EXPECT_EQ(alignof(gef::internal::RLE_GEF<int64_t>), 64u);
    EXPECT_LE(offset(&codec.run_rank) + sizeof(codec.run_rank), 64u);
    EXPECT_LE(offset(&codec.low_words) + sizeof(codec.low_words), 64u);
    EXPECT_LE(offset(&codec.high_words) + sizeof(codec.high_words), 64u);
    EXPECT_LE(offset(&codec.base) + sizeof(codec.base), 64u);
    EXPECT_LE(offset(&codec.b) + 1, 64u);
    EXPECT_LE(offset(&codec.h) + 1, 64u);
}
