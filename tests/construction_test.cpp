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

