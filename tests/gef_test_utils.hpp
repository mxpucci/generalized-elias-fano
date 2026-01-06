//
// Created by Michelangelo Pucci on 06/07/25.
//

#ifndef GEF_TEST_UTILS_HPP
#define GEF_TEST_UTILS_HPP

#include <gtest/gtest.h>
#include "gef/IGEF.hpp"
#include <vector>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>

// Helper trait to extract the underlying value_type from a GEF implementation class.
// Defined here to avoid redefinition errors in Unity Builds.
template<typename GEF>
struct get_value_type {
    using type = std::decay_t<decltype(std::declval<const GEF&>()[0])>;
};

namespace gef::test {

/**
 * @brief Generates a comprehensive random sequence with fine-grained controls.
 * @tparam T An integral type.
 * @param size The total number of elements in the sequence.
 * @param min_val The minimum possible value for an element.
 * @param max_val The maximum possible value for an element.
 * @param duplicate_chance The probability (0.0 to 1.0) that an element will be a duplicate of a previous one.
 * @param max_consecutive_duplicates The maximum number of times a value can be repeated consecutively.
 * @return A std::vector<T> with the generated sequence.
 */

#if __cplusplus >= 202002L
template<std::integral T>
#else
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
#endif
std::vector<T> generate_random_sequence(
    size_t size,
    T min_val,
    T max_val,
    double duplicate_chance = 0.0,
    int max_consecutive_duplicates = 1) {
    if (size == 0) {
        return {};
    }

    std::vector<T> sequence;
    sequence.reserve(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> value_dist(min_val, max_val);
    std::uniform_real_distribution<> duplicate_dist(0.0, 1.0);

    // Generate the first element
    sequence.push_back(value_dist(gen));

    int consecutive_count = 1;
    for (size_t i = 1; i < size; ++i) {
        // Decide whether to create a duplicate
        if (!sequence.empty() && duplicate_dist(gen) < duplicate_chance) {
            // Try to make a consecutive duplicate first
            if (consecutive_count < max_consecutive_duplicates) {
                sequence.push_back(sequence.back());
                consecutive_count++;
            } else {
                // Pick a random, non-consecutive duplicate from the existing sequence
                std::uniform_int_distribution<size_t> index_dist(0, sequence.size() - 1);
                size_t random_index = index_dist(gen);
                sequence.push_back(sequence[random_index]);
                if (sequence.back() == sequence[sequence.size() - 2]) {
                     consecutive_count++;
                } else {
                     consecutive_count = 1;
                }
            }
        } else {
            // Generate a new random value
            sequence.push_back(value_dist(gen));
            consecutive_count = 1;
        }
    }
    return sequence;
}

} // namespace gef::test

#endif // GEF_TEST_UTILS_HPP
