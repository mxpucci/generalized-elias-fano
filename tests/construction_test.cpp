#include <gtest/gtest.h>
#include "gef/B_STAR_GEF.hpp"
#include "gef/RLE_GEF.hpp"
#include "gef/gef.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
template<typename T>
void check_bstar_round_trip(const std::vector<T>& input, const gef::BlockParameters<T>* parameters,
                            gef::SplitPointStrategy strategy, const char* label) {
    using Codec = gef::internal::B_STAR_GEF<T, PastaGapBitVector, true>;
    const Codec codec = parameters ? Codec(input, *parameters) : Codec(input, strategy);
    ASSERT_EQ(codec.size(), input.size()) << label;
    std::vector<T> output(input.size());
    ASSERT_EQ(codec.get_elements(0, input.size(), output), input.size()) << label;
    ASSERT_EQ(output, input) << label;
    for (size_t i = 0; i < input.size(); ++i) ASSERT_EQ(codec[i], input[i]) << label << ' ' << i;
}

// Residuals mixing repeats, small steps of both signs and jumps longer than
// one or several words of unary code.
template<typename T>
std::vector<T> mixed_input(size_t length, unsigned width, std::mt19937_64& rng) {
    using U = std::make_unsigned_t<T>;
    const U mask = width == sizeof(T) * 8 ? U(~U(0)) : U((U(1) << width) - 1);
    std::vector<T> input(length);
    U current = U(rng()) & mask;
    for (size_t i = 0; i < length; ++i) {
        switch (rng() % 6) {
            case 0: break;
            case 1: current = U(current + rng() % 3) & mask; break;
            case 2: current = U(current - rng() % 3) & mask; break;
            case 3: current = U(current + (rng() % 1000) * 70) & mask; break;
            case 4: current = U(current - (rng() % 1000) * 70) & mask; break;
            default: current = U(rng()) & mask; break;
        }
        input[i] = static_cast<T>(current);
    }
    return input;
}

template<typename T>
void check_bstar_every_split() {
    std::mt19937_64 rng(9001 + sizeof(T));
    for (unsigned width : {1u, 5u, 13u, 20u, 31u, unsigned(sizeof(T) * 8 - 1), unsigned(sizeof(T) * 8)}) {
        // Forced parameters use base 0, so signed inputs stay non-negative.
        if (std::is_signed_v<T> && width == sizeof(T) * 8) continue;
        for (size_t length : {size_t(1), size_t(2), size_t(63), size_t(64), size_t(65), size_t(129), size_t(1000)}) {
            std::vector<T> input = mixed_input<T>(length, width, rng);
            // Make the stored base 0 and the width exact for forced parameters.
            input[0] = 0;
            for (unsigned split = 0; split <= width; ++split) {
                // Gaps reach 2^(width - split) bits of unary code; up to 2^16
                // covers codes spanning many words while keeping G small.
                if (width - split > 16) continue;
                const gef::BlockParameters<T> parameters{T(0), uint8_t(width), uint8_t(split)};
                check_bstar_round_trip(input, &parameters, gef::APPROXIMATE_SPLIT_POINT, "forced");
            }
            check_bstar_round_trip<T>(input, nullptr, gef::APPROXIMATE_SPLIT_POINT, "approximate");
            check_bstar_round_trip<T>(input, nullptr, gef::OPTIMAL_SPLIT_POINT, "optimal");
        }
    }
}
}

TEST(Construction, BStarRoundTripEverySplitInt64) { check_bstar_every_split<int64_t>(); }
TEST(Construction, BStarRoundTripEverySplitInt32) { check_bstar_every_split<int32_t>(); }
TEST(Construction, BStarRoundTripEverySplitUint64) { check_bstar_every_split<uint64_t>(); }

TEST(Construction, BStarPartitionedMatchesInput) {
    std::mt19937_64 rng(31337);
    std::vector<int64_t> input = mixed_input<int64_t>(100003, 40, rng);
    for (auto strategy : {gef::APPROXIMATE_SPLIT_POINT, gef::OPTIMAL_SPLIT_POINT}) {
        gef::B_STAR_GEF<int64_t> codec(input, strategy);
        std::vector<int64_t> output(input.size());
        ASSERT_EQ(codec.get_elements(0, input.size(), output), input.size());
        ASSERT_EQ(output, input);
    }
}

namespace {
// Smallest total size over every split; ties keep the larger split, as the
// encoder scans from b = total_bits downwards and keeps strict improvements.
template<typename T>
std::pair<unsigned, size_t> brute_force_rle_split(const std::vector<T>& input, unsigned total_bits, T base) {
    using U = std::make_unsigned_t<T>;
    unsigned best_b = total_bits;
    size_t best_space = size_t(total_bits) * input.size(), best_runs = 0;
    for (unsigned h = 1; h <= total_bits; ++h) {
        const unsigned b = total_bits - h;
        size_t runs = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            const U high = U(U(input[i]) - U(base)) >> b;
            runs += i == 0 || high != (U(U(input[i - 1]) - U(base)) >> b);
        }
        const size_t space = input.size() * (b + 1) + runs * h;
        if (space < best_space) { best_space = space; best_b = b; best_runs = runs; }
    }
    return {best_b, best_runs};
}

template<typename T>
void check_rle_split_choice() {
    using U = std::make_unsigned_t<T>;
    using Codec = gef::internal::RLE_GEF<T>;
    std::mt19937_64 rng(5150 + sizeof(T));
    constexpr unsigned W = sizeof(T) * 8;
    for (unsigned width : {1u, 3u, 9u, 17u, 30u, W - 1, W}) {
        for (size_t length : {size_t(1), size_t(2), size_t(3), size_t(5), size_t(64), size_t(65), size_t(200), size_t(3001)}) {
            for (int pattern = 0; pattern < 4; ++pattern) {
                const U mask = width == W ? U(~U(0)) : U((U(1) << width) - 1);
                std::vector<T> input(length);
                U current = U(rng()) & mask;
                for (size_t i = 0; i < length; ++i) {
                    // 0: long equal runs, 1: small steps, 2: random, 3: mixed.
                    const int kind = pattern == 3 ? int(rng() % 3) : pattern;
                    if (kind == 0 && rng() % 8 == 0) current = U(rng()) & mask;
                    if (kind == 1) current = U(current + rng() % 5 - 2) & mask;
                    if (kind == 2) current = U(rng()) & mask;
                    input[i] = static_cast<T>(current);
                }
                const Codec codec(input);
                T base = input[0];
                for (const T v : input) base = std::min(base, v);
                T top = input[0];
                for (const T v : input) top = std::max(top, v);
                const unsigned total_bits = std::bit_width(U(U(top) - U(base)));
                if (total_bits == 0) { EXPECT_EQ(codec.b + codec.h, 0); continue; }
                const auto [b, runs] = brute_force_rle_split(input, total_bits, base);
                ASSERT_EQ(codec.b, b) << width << ',' << length << ',' << pattern;
                if (b < total_bits) ASSERT_EQ(codec.H.size(), runs) << width << ',' << length << ',' << pattern;
                std::vector<T> output(length);
                ASSERT_EQ(codec.get_elements(0, length, output), length);
                ASSERT_EQ(output, input);
                for (size_t i = 0; i < length; ++i) ASSERT_EQ(codec[i], input[i]);
            }
        }
    }
}
}

TEST(Construction, RleSplitMatchesBruteForceInt64) { check_rle_split_choice<int64_t>(); }
TEST(Construction, RleSplitMatchesBruteForceInt32) { check_rle_split_choice<int32_t>(); }
TEST(Construction, RleSplitMatchesBruteForceUint64) { check_rle_split_choice<uint64_t>(); }
