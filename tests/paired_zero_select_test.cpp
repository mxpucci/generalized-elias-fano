#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include "datastructures/PairedZeroSelect.hpp"
#include "gef/B_STAR_GEF.hpp"
#include "gef/gef.hpp"

using gef::internal::PairedZeroSelect;

namespace {
// A gap stream as B* writes it: for each element, `gap` ones then one zero.
struct Stream {
    std::vector<uint64_t> words;   // exactly the words holding the bits
    std::vector<size_t> zeros;     // position of every zero
    size_t bits = 0;
};

Stream make_stream(const std::vector<size_t>& gaps) {
    Stream s;
    size_t total = gaps.size();
    for (size_t gap : gaps) total += gap;
    s.bits = total;
    s.words.assign((total + 63) / 64, 0);
    size_t position = 0;
    for (size_t gap : gaps) {
        for (size_t i = 0; i < gap; ++i, ++position) s.words[position / 64] |= uint64_t(1) << (position % 64);
        s.zeros.push_back(position++);
    }
    return s;
}

void check_directory(const Stream& plus, const Stream& minus, size_t claimed_extra_bits = 0) {
    ASSERT_EQ(plus.zeros.size(), minus.zeros.size());
    const size_t n = plus.zeros.size();
    ASSERT_TRUE(PairedZeroSelect::supports(n, plus.bits, minus.bits));
    PairedZeroSelect select;
    EXPECT_FALSE(select.enabled());
    // Reported sizes may exceed the stored words; only words up to the last
    // zero may be read (out-of-bounds reads are caught under ASan).
    select.build(plus.words.data(), plus.bits + claimed_extra_bits,
                 minus.words.data(), minus.bits + claimed_extra_bits, n);
    ASSERT_TRUE(select.enabled());
    EXPECT_GT(select.size_in_bytes(), 0u);
    for (size_t rank = 1; rank <= n; ++rank) {
        const auto pair = select.select0_pair(rank);
        ASSERT_EQ(pair.first, plus.zeros[rank - 1]) << "rank " << rank;
        ASSERT_EQ(pair.second, minus.zeros[rank - 1]) << "rank " << rank;
        ASSERT_EQ(select.select0(0, rank), plus.zeros[rank - 1]);
        ASSERT_EQ(select.select0(1, rank), minus.zeros[rank - 1]);
    }
    PairedZeroSelect moved(std::move(select));
    EXPECT_FALSE(select.enabled());
    ASSERT_TRUE(moved.enabled());
    for (size_t rank = 1; rank <= n; rank += std::max<size_t>(1, n / 97))
        ASSERT_EQ(moved.select0_pair(rank), std::make_pair(plus.zeros[rank - 1], minus.zeros[rank - 1]));
    PairedZeroSelect assigned;
    assigned = std::move(moved);
    EXPECT_FALSE(moved.enabled());
    ASSERT_EQ(assigned.select0_pair(n), std::make_pair(plus.zeros[n - 1], minus.zeros[n - 1]));
    assigned.clear();
    EXPECT_FALSE(assigned.enabled());
    EXPECT_EQ(assigned.size_in_bytes(), 0u);
}

std::vector<size_t> random_gaps(size_t n, size_t max_gap, double zero_probability, std::mt19937_64& rng) {
    std::vector<size_t> gaps(n);
    std::uniform_real_distribution<double> coin(0, 1);
    for (auto& gap : gaps) gap = coin(rng) < zero_probability ? 0 : rng() % (max_gap + 1);
    return gaps;
}

// Gaps for n zeros whose stream is exactly `bits` long and ends with a zero.
std::vector<size_t> gaps_for_length(size_t n, size_t bits) {
    std::vector<size_t> gaps(n, 0);
    size_t ones = bits - n;
    for (size_t i = 0; ones; i = (i + 1) % n) {
        const size_t add = std::min<size_t>(ones, 3);
        gaps[i] += add;
        ones -= add;
    }
    return gaps;
}
} // namespace

TEST(PairedZeroSelect, SingleZero) {
    for (size_t gap : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(511), size_t(512), size_t(70000)})
        check_directory(make_stream({gap}), make_stream({0}));
}

TEST(PairedZeroSelect, AllZerosExhaustive) {
    for (size_t n : {size_t(1), size_t(2), size_t(63), size_t(64), size_t(65), size_t(511), size_t(512),
                     size_t(513), size_t(16384), size_t(16385), size_t(65535)}) {
        SCOPED_TRACE(n);
        const auto zeros = make_stream(std::vector<size_t>(n, 0));
        check_directory(zeros, zeros);
    }
}

TEST(PairedZeroSelect, SmallStreamsExhaustiveRanks) {
    std::mt19937_64 rng(1203);
    for (size_t n = 1; n <= 300; ++n) {
        SCOPED_TRACE(n);
        check_directory(make_stream(random_gaps(n, 9, 0.5, rng)), make_stream(random_gaps(n, 70, 0.8, rng)));
    }
}

TEST(PairedZeroSelect, LongRunsOfOnesSpanManyChunks) {
    std::mt19937_64 rng(77);
    // Up to ~20 directory chunks per side; also one huge run at either end.
    check_directory(make_stream(random_gaps(200, 4000, 0.2, rng)), make_stream(random_gaps(200, 3, 0.5, rng)));
    std::vector<size_t> front(1000, 0), back(1000, 0);
    front.front() = 300000;
    back.back() = 300000;
    check_directory(make_stream(front), make_stream(back));
    check_directory(make_stream(back), make_stream(front));
}

TEST(PairedZeroSelect, ExactChunkAndBlockBoundaries) {
    // 512 bits per block, 32 blocks per chunk; also one bit more or less.
    for (size_t blocks : {size_t(1), size_t(31), size_t(32), size_t(33), size_t(64), size_t(128), size_t(129)}) {
        for (long delta : {-1L, 0L, 1L}) {
            const size_t bits = blocks * 512 + delta;
            const size_t n = bits / 3;
            SCOPED_TRACE(::testing::Message() << "bits=" << bits);
            const auto exact = make_stream(gaps_for_length(n, bits));
            ASSERT_EQ(exact.bits, bits);
            check_directory(exact, make_stream(std::vector<size_t>(n, 0)));
            check_directory(make_stream(std::vector<size_t>(n, 1)), exact);
        }
    }
}

TEST(PairedZeroSelect, ReportedSizeBeyondIndexedWords) {
    std::mt19937_64 rng(5);
    for (size_t n : {size_t(1), size_t(100), size_t(5000)})
        check_directory(make_stream(random_gaps(n, 5, 0.5, rng)), make_stream(random_gaps(n, 5, 0.5, rng)), 1 << 20);
}

TEST(PairedZeroSelect, RandomStreams) {
    std::mt19937_64 rng(90210);
    for (int round = 0; round < 40; ++round) {
        const size_t n = 1 + rng() % 65535;
        const size_t max_gap = size_t(1) << (rng() % 8);
        SCOPED_TRACE(::testing::Message() << "n=" << n << " max_gap=" << max_gap);
        check_directory(make_stream(random_gaps(n, max_gap, 0.6, rng)), make_stream(random_gaps(n, max_gap, 0.6, rng)));
    }
}

TEST(PairedZeroSelect, SupportsLimit) {
    EXPECT_TRUE(PairedZeroSelect::supports(PairedZeroSelect::MAX_ZEROS, 1 << 20, 1 << 20));
    EXPECT_FALSE(PairedZeroSelect::supports(PairedZeroSelect::MAX_ZEROS + 1, 1 << 20, 1 << 20));
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
    Codec other(std::vector<T>{T(1), T(2), T(3)});
    swap(other, move_assigned);
    expect_decodes(other, values);
    expect_decodes(move_assigned, std::vector<T>{T(1), T(2), T(3)});

    const auto path = std::filesystem::temp_directory_path() /
                      ("paired_zero_select_test_" + std::to_string(::getpid()) + ".bin");
    codec.serialize(path);
    Codec loaded;
    loaded.load(path);
    std::filesystem::remove(path);
    expect_decodes(loaded, values);
    EXPECT_EQ(loaded.size_in_bytes(), codec.size_in_bytes());
}

template<class T>
std::vector<T> random_walk(size_t n, int64_t start, int64_t step, std::mt19937_64& rng) {
    std::vector<T> values(n);
    int64_t value = start;
    for (auto& x : values) {
        value += static_cast<int64_t>(rng() % (2 * step + 1)) - step;
        x = static_cast<T>(value);
    }
    return values;
}
} // namespace

// Sizes above PairedZeroSelect::MAX_ZEROS keep Pasta select0; smaller ones
// use the paired directory. Both must decode identically through every
// lifecycle operation.
constexpr size_t CODEC_SIZES[] = {1, 2, 64, 513, 4097, 32000, 65535, 65536, 70001};

TEST(PairedZeroSelectBStar, SignedInt64RandomWalks) {
    std::mt19937_64 rng(4242);
    for (size_t n : CODEC_SIZES) {
        SCOPED_TRACE(n);
        const auto values = random_walk<int64_t>(n, -5000, 900, rng);
        check_codec_lifecycle(gef::internal::B_STAR_GEF<int64_t>(values, gef::OPTIMAL_SPLIT_POINT), values);
        check_codec_lifecycle(gef::internal::B_STAR_GEF<int64_t>(values, gef::APPROXIMATE_SPLIT_POINT), values);
    }
}

TEST(PairedZeroSelectBStar, Int32AndUint64) {
    std::mt19937_64 rng(31337);
    for (size_t n : CODEC_SIZES) {
        SCOPED_TRACE(n);
        const auto small = random_walk<int32_t>(n, -100000, 3000, rng);
        check_codec_lifecycle(gef::internal::B_STAR_GEF<int32_t>(small, gef::OPTIMAL_SPLIT_POINT), small);
        std::vector<uint64_t> large(n);
        for (auto& x : large) x = UINT64_MAX - (rng() % (uint64_t(1) << 40));
        check_codec_lifecycle(gef::internal::B_STAR_GEF<uint64_t>(large, gef::OPTIMAL_SPLIT_POINT), large);
    }
}

TEST(PairedZeroSelectBStar, ForcedSplitsIncludingNoLowAndNoHighPart) {
    std::mt19937_64 rng(8);
    for (size_t n : {size_t(1), size_t(700), size_t(40000), size_t(66000)}) {
        std::vector<int64_t> values(n);
        for (auto& x : values) x = -2048 + static_cast<int64_t>(rng() % 4096);
        for (uint8_t split : {uint8_t(0), uint8_t(1), uint8_t(7), uint8_t(11), uint8_t(12)}) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " split=" << int(split));
            // split 0: no low part (b == 0); split 12 == width: no high part (h == 0).
            const gef::BlockParameters<int64_t> parameters{-2048, 12, split};
            check_codec_lifecycle(gef::internal::B_STAR_GEF<int64_t>(values, parameters), values);
        }
    }
    const std::vector<int64_t> constant(5000, -77);
    check_codec_lifecycle(gef::internal::B_STAR_GEF<int64_t>(constant, gef::OPTIMAL_SPLIT_POINT), constant);
}

TEST(PairedZeroSelectBStar, PartitionedCodecs) {
    std::mt19937_64 rng(2718);
    const auto values = random_walk<int64_t>(250000, 0, 5000, rng);
    // 100000-element partitions exceed MAX_ZEROS and use the Pasta fallback.
    expect_decodes(gef::B_STAR_GEF<int64_t>(values), values);
    expect_decodes(gef::B_STAR_GEF<int64_t, 100000>(values), values);
    expect_decodes(gef::PARTITIONED_B_STAR_GEF<int64_t>(values), values);
}

TEST(PairedZeroSelectBStar, LoadsStreamsWithSerializedPastaSupport) {
    // Earlier versions enabled Pasta select0 on every gap vector, so their
    // streams set the per-bitvector support flag. Loading such a stream must
    // decode identically and must not keep (or count) the unused support.
    std::mt19937_64 rng(606);
    const auto values = random_walk<int64_t>(20000, 0, 40, rng);
    const int64_t base = *std::min_element(values.begin(), values.end());
    // b == 0: the stream is h, b, size, base, then G+ and G- (no L).
    const gef::BlockParameters<int64_t> parameters{base, 12, 0};
    for (auto v : values) ASSERT_LT(v - base, 4096);
    gef::internal::B_STAR_GEF<int64_t> codec(values, parameters);
    const auto path = std::filesystem::temp_directory_path() /
                      ("paired_zero_select_flag_" + std::to_string(::getpid()) + ".bin");
    codec.serialize(path);
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    size_t offset = 2 + sizeof(size_t) + sizeof(int64_t);
    for (int side = 0; side < 2; ++side) {
        size_t bits = 0;
        file.seekg(offset);
        file.read(reinterpret_cast<char*>(&bits), sizeof(bits));
        file.seekp(offset + sizeof(bits));
        const bool has_support = true;
        file.write(reinterpret_cast<const char*>(&has_support), sizeof(has_support));
        offset += sizeof(bits) + sizeof(bool) + ((bits >> 6) + 1) * sizeof(uint64_t);
    }
    file.close();
    ASSERT_EQ(std::filesystem::file_size(path), offset);
    gef::internal::B_STAR_GEF<int64_t> loaded;
    loaded.load(path);
    std::filesystem::remove(path);
    expect_decodes(loaded, values);
    EXPECT_EQ(loaded.size_in_bytes(), codec.size_in_bytes());
}
