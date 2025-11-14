#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "gef/gap_computation_utils.hpp"

namespace gef::test {

inline size_t bits_to_bytes(size_t bits) {
    return (bits + 7) / 8;
}

template<typename T>
size_t compute_total_bits_from_range(T min_val, T max_val) {
    using WI = __int128;
    using WU = unsigned __int128;
    const WI min_w = static_cast<WI>(min_val);
    const WI max_w = static_cast<WI>(max_val);
    const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);

    size_t bits = 1;
    if (range > 1) {
        bits = 0;
        WU x = range - 1;
        while (x > 0) {
            ++bits;
            x >>= 1;
        }
    }

    const size_t width = sizeof(T) * 8;
    return std::min<size_t>(bits, width);
}

template<typename T>
size_t theoretical_size_for_split(const std::vector<T>& sequence,
                                  T min_val,
                                  T max_val,
                                  size_t total_bits,
                                  uint8_t requested_b) {
    if (sequence.empty()) {
        return sizeof(T) + sizeof(uint8_t) * 2;
    }

    const uint8_t b = static_cast<uint8_t>(std::min<size_t>(requested_b, total_bits));
    size_t total_bytes = sizeof(T) + sizeof(uint8_t) * 2;
    total_bytes += bits_to_bytes(sequence.size() * static_cast<size_t>(b));

    if (total_bits > b) {
        const auto gaps = variation_of_shifted_vec(sequence, min_val, max_val, b, ExceptionRule::None);
        total_bytes += bits_to_bytes(gaps.sum_of_positive_gaps + sequence.size());
        total_bytes += bits_to_bytes(gaps.sum_of_negative_gaps + sequence.size());
    }

    return total_bytes;
}

} // namespace gef::test

