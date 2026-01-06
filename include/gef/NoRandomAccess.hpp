#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace gef::internal::no_support {

// Decode-prefix callback concept:
//   size_t decode_prefix(size_t n, std::vector<T>& out)
// It must write the first min(n, size()) decoded values into out[0..).

template<typename T, typename DecodePrefixFn>
size_t get_elements_force_from_zero(size_t startIndex,
                                    size_t count,
                                    size_t total_size,
                                    std::vector<T>& output,
                                    DecodePrefixFn&& decode_prefix) {
    if (count == 0 || startIndex >= total_size) {
        return 0;
    }
    if (output.size() < count) {
        throw std::invalid_argument("output buffer is smaller than requested count");
    }

    const size_t total = std::min(startIndex + count, total_size);

    if (startIndex == 0) {
        // total <= count, so output has enough capacity.
        return decode_prefix(total, output);
    }

    std::vector<T> tmp(total);
    const size_t written_total = decode_prefix(total, tmp);
    if (written_total <= startIndex) {
        return 0;
    }
    const size_t available = written_total - startIndex;
    const size_t to_copy = std::min(count, available);
    for (size_t i = 0; i < to_copy; ++i) {
        output[i] = tmp[startIndex + i];
    }
    return to_copy;
}

template<typename T, typename DecodePrefixFn>
T at_force_from_zero(size_t index,
                     std::vector<T>& scratch,
                     DecodePrefixFn&& decode_prefix) {
    scratch.resize(index + 1);
    (void)decode_prefix(index + 1, scratch);
    return scratch[index];
}

} // namespace gef::internal::no_support


