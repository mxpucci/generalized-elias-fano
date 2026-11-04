#include <gtest/gtest.h>

#include "gef/gef.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace {

using Wide = unsigned __int128;

unsigned width_of(uint64_t value) {
    unsigned width = 0;
    while (value) {
        ++width;
        value >>= 1;
    }
    return width;
}

uint64_t high(uint64_t value, unsigned split) {
    return split == 64 ? 0 : value >> split;
}

template<class T>
uint64_t difference(T value, T base) {
    return static_cast<uint64_t>(static_cast<__int128>(value) - static_cast<__int128>(base));
}

uint64_t distance(uint64_t left, uint64_t right) {
    return left > right ? left - right : right - left;
}

template<class T>
Wide bstar_payload(const std::vector<T>& values, size_t begin, size_t end,
                   T base, unsigned width, unsigned split) {
    const size_t count = end - begin;
    if (split == width) return Wide(count) * width;
    Wide bits = Wide(count) * (split + 2);
    uint64_t previous = 0;
    for (size_t i = begin; i < end; ++i) {
        const uint64_t current = high(difference(values[i], base), split);
        bits += distance(current, previous);
        previous = current;
    }
    return bits;
}

template<class T>
Wide bound_payload(const std::vector<T>& values, size_t begin, size_t end,
                   T base, unsigned width, unsigned split) {
    if (split == width) return Wide(end - begin) * width;
    Wide variation = difference(values[begin], base);
    for (size_t i = begin + 1; i < end; ++i) {
        variation += distance(difference(values[i], base), difference(values[i - 1], base));
    }
    const Wide divisor = Wide(1) << split;
    return Wide(end - begin) * (split + 3) + (variation + divisor - 1) / divisor;
}

template<class T>
Wide rle_payload(const std::vector<T>& values, size_t begin, size_t end,
                 T base, unsigned width, unsigned split) {
    if (split == width) return Wide(end - begin) * width;
    size_t runs = 1;
    for (size_t i = begin + 1; i < end; ++i) {
        runs += high(difference(values[i], base), split) !=
                high(difference(values[i - 1], base), split);
    }
    return Wide(end - begin) * (split + 1) + Wide(width - split) * runs;
}

template<class T>
void check_oracles(const std::vector<T>& values) {
    gef::PartitionOptions options;
    options.grid = 1;
    options.block_overhead_bits = 17;
    gef::ExactBStarOracle<T> exact(values, options);
    gef::BoundBStarOracle<T> bound(values, options);
    gef::RLEPartitionOracle<T> rle(values, options);
    const T base = *std::min_element(values.begin(), values.end());
    const unsigned width = width_of(difference(*std::max_element(values.begin(), values.end()), base));
    for (size_t begin = 0; begin < values.size(); ++begin) {
        for (size_t end = begin + 1; end <= values.size(); ++end) {
            SCOPED_TRACE(::testing::Message() << "interval=" << begin << ',' << end);
            Wide best_exact = Wide(end - begin) * width;
            Wide best_bound = best_exact;
            for (unsigned split = 0; split < width; ++split) {
                best_exact = std::min(best_exact, bstar_payload(values, begin, end, base, width, split));
                best_bound = std::min(best_bound, bound_payload(values, begin, end, base, width, split));
            }
            const auto actual = exact(begin, end);
            const auto upper = bound(begin, end);
            EXPECT_EQ(actual.bits, uint64_t(best_exact) + options.block_overhead_bits);
            EXPECT_EQ(upper.bits, uint64_t(best_bound) + options.block_overhead_bits);
            EXPECT_GE(upper.bits, actual.bits);
            EXPECT_LE(upper.bits - actual.bits, 2 * (end - begin));
            EXPECT_EQ(actual.parameters.base, base);
            EXPECT_EQ(actual.parameters.width, width);
            EXPECT_EQ(actual.bits, uint64_t(bstar_payload(values, begin, end, base, width,
                                                         actual.parameters.split)) + options.block_overhead_bits);
            EXPECT_EQ(upper.bits, uint64_t(bound_payload(values, begin, end, base, width,
                                                        upper.parameters.split)) + options.block_overhead_bits);
            unsigned local_width = 0;
            for (size_t i = begin + 1; i < end; ++i) {
                local_width = std::max(local_width, width_of(difference(values[i], base) ^
                                                              difference(values[i - 1], base)));
            }
            const uint64_t anchor = local_width == 64 ? 0 :
                                    (difference(values[begin], base) >> local_width) << local_width;
            const T local_base = static_cast<T>(static_cast<__int128>(base) + anchor);
            Wide best_rle = Wide(end - begin) * local_width;
            for (unsigned split = 0; split < local_width; ++split) {
                best_rle = std::min(best_rle, rle_payload(values, begin, end, local_base, local_width, split));
            }
            const auto runs = rle(begin, end);
            EXPECT_EQ(runs.bits, uint64_t(best_rle) + options.block_overhead_bits);
            EXPECT_EQ(runs.parameters.base, local_base);
            EXPECT_EQ(runs.parameters.width, local_width);
            EXPECT_EQ(runs.bits, uint64_t(rle_payload(values, begin, end, local_base, local_width,
                                                     runs.parameters.split)) + options.block_overhead_bits);
            if (end - begin > 1) {
                EXPECT_LE(exact(begin + 1, end).bits, actual.bits);
                EXPECT_LE(exact(begin, end - 1).bits, actual.bits);
                EXPECT_LE(bound(begin + 1, end).bits, upper.bits);
                EXPECT_LE(bound(begin, end - 1).bits, upper.bits);
                EXPECT_LE(rle(begin + 1, end).bits, runs.bits);
                EXPECT_LE(rle(begin, end - 1).bits, runs.bits);
            }
        }
    }
}

TEST(PartitionOracles, ExhaustiveSplitsAndMonotoneIntervals) {
    check_oracles<int64_t>({42});
    check_oracles<int64_t>({-7, -7, -7, -7});
    check_oracles<int64_t>({-20, 200, 199, -18, -19, 0, 0, 0, 31, 32, 63, 64});
    check_oracles<uint64_t>({0, UINT64_MAX, UINT64_MAX - 1, 1, uint64_t(1) << 63});
    check_oracles<int64_t>({INT64_MIN, INT64_MAX, INT64_MIN + 1, 0, -1});
    std::vector<uint64_t> plateau(32, UINT64_MAX - 3);
    plateau[0] = 0;
    check_oracles(plateau);
    std::mt19937_64 random(2194);
    for (unsigned sample = 0; sample < 12; ++sample) {
        std::vector<int64_t> values(24);
        for (auto& value : values) value = int64_t(random() % 1000000) - 500000;
        check_oracles(values);
        int64_t previous = 0;
        for (auto& value : values) {
            previous += int64_t(random() % 7) - 3;
            value = previous;
        }
        values.back() = 1000000;
        check_oracles(values);
    }
}

template<class Oracle>
void check_partition_plan(const Oracle& oracle, gef::PartitionOptions options) {
    std::vector<size_t> endpoints{0};
    for (size_t i = options.grid; i < oracle.size(); i += options.grid) endpoints.push_back(i);
    if (oracle.size()) endpoints.push_back(oracle.size());
    std::vector<uint64_t> optimal(endpoints.size(), std::numeric_limits<uint64_t>::max());
    optimal[0] = 0;
    for (size_t right = 1; right < endpoints.size(); ++right) {
        for (size_t left = 0; left < right; ++left) {
            optimal[right] = std::min(optimal[right], optimal[left] + oracle(endpoints[left], endpoints[right]).bits);
        }
    }
    const auto exact = gef::partition_sequence(oracle, options, true);
    const auto approximate = gef::partition_sequence(oracle, options);
    EXPECT_EQ(exact.estimated_bits, optimal.back());
    EXPECT_LE(static_cast<long double>(approximate.estimated_bits),
              (1 + options.epsilon) * static_cast<long double>(optimal.back()));
    for (const auto* plan : {&exact, &approximate}) {
        uint64_t cost = 0;
        size_t begin = 0;
        for (size_t end : plan->ends) {
            EXPECT_GT(end, begin);
            EXPECT_TRUE(end == oracle.size() || end % options.grid == 0);
            cost += oracle(begin, end).bits;
            begin = end;
        }
        EXPECT_EQ(begin, oracle.size());
        EXPECT_EQ(cost, plan->estimated_bits);
    }
}

TEST(PartitionAlgorithm, MatchesIndependentDynamicProgram) {
    std::mt19937_64 random(313);
    for (unsigned sample = 0; sample < 12; ++sample) {
        std::vector<int64_t> values(43);
        for (size_t i = 0; i < values.size(); ++i) values[i] = (i / 7) % 2 ? int64_t(random() % 10000) : 5;
        for (size_t grid : {size_t(1), size_t(3), size_t(16)}) {
            gef::PartitionOptions options;
            options.grid = grid;
            options.epsilon = 0.2;
            options.block_overhead_bits = 9;
            check_partition_plan(gef::ExactBStarOracle<int64_t>(values, options), options);
            check_partition_plan(gef::BoundBStarOracle<int64_t>(values, options), options);
            check_partition_plan(gef::RLEPartitionOracle<int64_t>(values, options), options);
        }
    }
    gef::PartitionOptions options;
    const std::vector<int64_t> empty;
    check_partition_plan(gef::ExactBStarOracle<int64_t>(empty, options), options);
    check_partition_plan(gef::BoundBStarOracle<int64_t>(empty, options), options);
    check_partition_plan(gef::RLEPartitionOracle<int64_t>(empty, options), options);
}

template<class Index, class T>
void check_decoding(const Index& index, const std::vector<T>& values) {
    ASSERT_EQ(index.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) EXPECT_EQ(index[i], values[i]) << i;
    std::vector<T> output(values.size() + 5);
    for (size_t begin = 0; begin <= values.size(); ++begin) {
        const size_t count = index.get_elements(begin, output.size(), output);
        ASSERT_EQ(count, values.size() - begin);
        EXPECT_TRUE(std::equal(output.begin(), output.begin() + count, values.begin() + begin));
    }
    EXPECT_THROW(index.at(values.size()), std::out_of_range);
}

template<class Index, class T>
void check_round_trip(const std::vector<T>& values, gef::PartitionOptions options) {
    Index original(values, options);
    check_decoding(original, values);
    if (!values.empty()) EXPECT_EQ(original.partition_ends().back(), values.size());
    Index moved(std::move(original));
    check_decoding(moved, values);
    const auto path = std::filesystem::temp_directory_path() / "gef_optimal_partitioning_test.bin";
    {
        std::ofstream stream(path, std::ios::binary);
        moved.serialize(stream);
    }
    Index restored;
    {
        std::ifstream stream(path, std::ios::binary);
        restored.load(stream);
    }
    std::filesystem::remove(path);
    check_decoding(restored, values);
    EXPECT_EQ(restored.partition_ends(), moved.partition_ends());
    EXPECT_EQ(restored.estimated_bits(), moved.estimated_bits());
}

template<class T>
void check_public_type() {
    gef::PartitionOptions options;
    options.grid = 1;
    options.block_overhead_bits = 8;
    std::vector<T> values;
    for (size_t i = 0; i < 65; ++i) values.push_back(T(i < 20 ? 10 : i < 45 ? (i * 131) % 1000 : 900));
    if constexpr (std::is_signed_v<T>) for (T& value : values) value -= 500;
    check_round_trip<gef::PARTITIONED_B_STAR_GEF<T>>(values, options);
    check_round_trip<gef::PARTITIONED_B_STAR_GEF<T, gef::BoundBStarOracle<T>>>(values, options);
    check_round_trip<gef::PARTITIONED_B_STAR_GEF<T, gef::ExactBStarOracle<T>, false>>(values, options);
    check_round_trip<gef::PARTITIONED_RLE_GEF<T>>(values, options);
    check_round_trip<gef::PARTITIONED_RLE_GEF<T, false>>(values, options);
    check_round_trip<gef::PARTITIONED_B_STAR_GEF<T>>(std::vector<T>(12, T(42)), options);
    check_round_trip<gef::PARTITIONED_RLE_GEF<T>>(std::vector<T>(12, T(42)), options);
    check_round_trip<gef::PARTITIONED_B_STAR_GEF<T>>(std::vector<T>{}, options);
    check_round_trip<gef::PARTITIONED_RLE_GEF<T>>(std::vector<T>{}, options);
    const T maximum = std::numeric_limits<T>::max();
    const T minimum = std::numeric_limits<T>::min();
    for (const auto& edge : {std::vector<T>{T(maximum - 3), maximum, T(maximum - 1), maximum},
                             std::vector<T>{minimum, T(minimum + 3), T(minimum + 1), minimum}}) {
        check_round_trip<gef::PARTITIONED_B_STAR_GEF<T>>(edge, options);
        check_round_trip<gef::PARTITIONED_RLE_GEF<T>>(edge, options);
    }
}

TEST(PartitionedGEF, PublicTypesAndSerialization) {
    check_public_type<int32_t>();
    check_public_type<uint32_t>();
    check_public_type<int64_t>();
    check_public_type<uint64_t>();
}

TEST(PartitionedGEF, FullWidthRoundTrips) {
    gef::PartitionOptions options;
    options.grid = 1;
    const std::vector<int64_t> signed_values{INT64_MIN, INT64_MAX, 0, -1, INT64_MIN + 1};
    const std::vector<uint64_t> unsigned_values{0, UINT64_MAX, 1, UINT64_MAX - 1, uint64_t(1) << 63};
    check_round_trip<gef::PARTITIONED_B_STAR_GEF<int64_t>>(signed_values, options);
    check_round_trip<gef::PARTITIONED_B_STAR_GEF<uint64_t>>(unsigned_values, options);
    check_round_trip<gef::PARTITIONED_RLE_GEF<int64_t>>(signed_values, options);
    check_round_trip<gef::PARTITIONED_RLE_GEF<uint64_t>>(unsigned_values, options);
}

TEST(PartitionAlgorithm, RejectsInvalidOptionsAndIntervals) {
    const std::vector<int64_t> values{1, 2, 3, 4, 5, 6, 7};
    gef::PartitionOptions options;
    options.grid = 0;
    EXPECT_THROW(gef::ExactBStarOracle<int64_t>(values, options), std::invalid_argument);
    EXPECT_THROW(gef::BoundBStarOracle<int64_t>(values, options), std::invalid_argument);
    EXPECT_THROW(gef::RLEPartitionOracle<int64_t>(values, options), std::invalid_argument);
    options.grid = 3;
    gef::ExactBStarOracle<int64_t> exact(values, options);
    gef::BoundBStarOracle<int64_t> bound(values, options);
    gef::RLEPartitionOracle<int64_t> rle(values, options);
    EXPECT_THROW(exact(1, 3), std::invalid_argument);
    EXPECT_THROW(bound(0, 2), std::invalid_argument);
    EXPECT_THROW(rle(1, 6), std::invalid_argument);
    EXPECT_THROW(exact(3, 3), std::out_of_range);
    EXPECT_THROW(bound(0, 8), std::out_of_range);
    EXPECT_THROW(rle(6, 3), std::out_of_range);
    for (double epsilon : {0.0, -0.1, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        options.epsilon = epsilon;
        EXPECT_THROW(gef::partition_sequence(exact, options), std::invalid_argument);
    }
    options.epsilon = 0.1;
    options.grid = 1;
    EXPECT_THROW(gef::partition_sequence(exact, options), std::invalid_argument);
}

TEST(PartitionedGEF, ExplicitNativeParametersAndWordBoundaries) {
    using BStar = gef::internal::B_STAR_GEF<int64_t>;
    using RLE = gef::internal::RLE_GEF<int64_t>;
    for (size_t count : {size_t(1), size_t(2), size_t(63), size_t(64), size_t(65)}) {
        std::vector<int64_t> values(count);
        for (size_t i = 0; i < count; ++i) values[i] = -100 + int64_t((i * 11) % 23);
        for (uint8_t split : {uint8_t(0), uint8_t(1), uint8_t(7), uint8_t(32), uint8_t(63), uint8_t(64)}) {
            const gef::BlockParameters<int64_t> parameters{-101, 64, split};
            BStar bstar(values, parameters);
            RLE rle(values, parameters);
            check_decoding(bstar, values);
            check_decoding(rle, values);
        }
    }
    const std::vector<int64_t> values{-10, -2, 0, 4};
    for (const auto& invalid : {gef::BlockParameters<int64_t>{0, 8, 3},
                                gef::BlockParameters<int64_t>{-10, 3, 1},
                                gef::BlockParameters<int64_t>{-10, 65, 1},
                                gef::BlockParameters<int64_t>{-10, 5, 6}}) {
        EXPECT_THROW(BStar(values, invalid), std::invalid_argument);
        EXPECT_THROW(RLE(values, invalid), std::invalid_argument);
    }
}

template<class Index>
void check_truncated_loads() {
    gef::PartitionOptions options;
    options.grid = 1;
    options.block_overhead_bits = 4096;
    std::vector<int64_t> values(64);
    for (size_t i = 0; i < values.size(); ++i) values[i] = i / 8;
    Index original(values, options);
    ASSERT_EQ(original.partition_ends().size(), 1u);
    const auto path = std::filesystem::temp_directory_path() / "gef_partition_truncation_test.bin";
    {
        std::ofstream stream(path, std::ios::binary);
        original.serialize(stream);
    }
    std::vector<char> bytes;
    {
        std::ifstream stream(path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    }
    const std::vector<int64_t> previous{100, -50, 9, 2};
    Index destination(previous, options);
    const auto previous_ends = destination.partition_ends();
    const auto previous_cost = destination.estimated_bits();
    const size_t directory = sizeof(size_t) * 2 + sizeof(uint64_t);
    for (size_t prefix : {size_t(0), sizeof(size_t) - 1, directory - 1, directory,
                          directory + 1, directory + 18, directory + 25,
                          (directory + bytes.size()) / 2, bytes.size() - 1}) {
        ASSERT_LT(prefix, bytes.size());
        {
            std::ofstream stream(path, std::ios::binary);
            stream.write(bytes.data(), prefix);
        }
        for (auto mask : {std::ios::goodbit, std::ios::failbit | std::ios::badbit}) {
            std::ifstream stream(path, std::ios::binary);
            stream.exceptions(mask);
            EXPECT_ANY_THROW(destination.load(stream)) << "prefix=" << prefix;
            EXPECT_EQ(stream.exceptions(), mask);
            EXPECT_EQ(destination.partition_ends(), previous_ends);
            EXPECT_EQ(destination.estimated_bits(), previous_cost);
            check_decoding(destination, previous);
        }
    }
    std::filesystem::remove(path);
}

TEST(PartitionedGEF, TruncatedLoadsPreserveIndexAndStreamPolicy) {
    check_truncated_loads<gef::PARTITIONED_B_STAR_GEF<int64_t>>();
    check_truncated_loads<gef::PARTITIONED_RLE_GEF<int64_t>>();
}

class CountingOracle {
    gef::ExactBStarOracle<int64_t> oracle_;
    size_t* calls_;

public:
    using value_type = int64_t;
    CountingOracle(const std::vector<int64_t>& values, gef::PartitionOptions options, size_t& calls)
        : oracle_(values, options), calls_(&calls) {}
    size_t size() const { return oracle_.size(); }
    size_t grid() const { return oracle_.grid(); }
    uint64_t minimum_block_cost() const { return oracle_.minimum_block_cost(); }
    auto operator()(size_t begin, size_t end) const {
        ++*calls_;
        return oracle_(begin, end);
    }
};

TEST(PartitionedGEF, AcceptsCustomOracleObject) {
    const std::vector<int64_t> values{9, 8, 7, 30, 30, 30, 0, -1, 0, 400};
    gef::PartitionOptions options;
    options.grid = 1;
    size_t calls = 0;
    CountingOracle oracle(values, options, calls);
    gef::PARTITIONED_B_STAR_GEF<int64_t, CountingOracle> index(values, oracle, options);
    EXPECT_GT(calls, 0u);
    check_decoding(index, values);
}

}
