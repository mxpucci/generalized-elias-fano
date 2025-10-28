//
// Created by Michelangelo Pucci on 07/10/25.
//

#ifndef UTILS_HPP
#define UTILS_HPP

#include "IGEF.hpp"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <bit>
#include <cmath>
#include <type_traits>

#ifdef __AVX2__
#include <immintrin.h>
#endif

using namespace gef;

enum class ExceptionRule : uint8_t {
    None  = 0,  // Rule #0  (B_GEF_STAR): never an exception
    BGEF  = 1,  // Rule #1  (B_GEF):      i==0 || abs(gap)+2 > total_bits - b
    UGEF  = 2   // Rule #2  (U_GEF):      i==0 || gap<0 || gap+1 > total_bits - b
};



/*
 * positive_gaps = # i s.t. s[i] >= s[i - 1]
 * negative_gaps = n - positive_gaps
 * positive_exceptions_count = negative_exceptions_count = 0
 * sum_of_positive_gaps = sum_of_positive_gaps_without_exception = \sum { s_i - s_{i - 1} : s[i] >= s[i - 1]}
 * sum_of_negative_gaps = sum_of_negative_gaps_without_exception = \sum { s_{i - 1} - s_i : s[i] < s[i - 1]}
 */
template<typename T>
GapComputation variation_of_original_vec(
    const std::vector<T>& vec,
    const T min_val,
    const T max_val) {
    GapComputation result{};
    size_t n = vec.size();
    if (n < 2) return result;

    size_t pos_gaps = 0;
    size_t sum_pos = 0;
    size_t sum_neg = 0;

    // first gap is always non-negative since min_val <= vec[0]
    {
        uint64_t a0 = static_cast<uint64_t>(vec[0]);
        uint64_t b0 = static_cast<uint64_t>(min_val);
        sum_pos += (a0 - b0);   // unsigned diff = correct magnitude
        ++pos_gaps;             // count the initial positive gap
    }

    // Optimized scalar loop - compiler can auto-vectorize this with -O3 -march=native
    // Manual SIMD adds complexity without much benefit for modern compilers
    for (size_t i = 1; i < n; ++i) {
        // Use signed arithmetic for correct handling of negative values
        const int64_t curr = static_cast<int64_t>(vec[i]) - static_cast<int64_t>(min_val);
        const int64_t prev = static_cast<int64_t>(vec[i - 1]) - static_cast<int64_t>(min_val);
        const int64_t diff = curr - prev;
        
        // Branchless computation - compiler can vectorize this
        const bool is_positive = diff >= 0;
        const uint64_t mag = is_positive ? static_cast<uint64_t>(diff) : static_cast<uint64_t>(-diff);
        
        pos_gaps += is_positive;
        sum_pos += is_positive ? mag : 0;
        sum_neg += is_positive ? 0 : mag;
    }

    result.positive_gaps = pos_gaps;
    result.negative_gaps = (n - 1) - pos_gaps;
    result.positive_exceptions_count = 0;
    result.negative_exceptions_count = 0;
    result.sum_of_positive_gaps = sum_pos;
    result.sum_of_negative_gaps = sum_neg;
    result.sum_of_positive_gaps_without_exception = sum_pos;
    result.sum_of_negative_gaps_without_exception = sum_neg;
    return result;
}

template<typename T>
GapComputation variation_of_shifted_vec(
    const std::vector<T>& v,
    const T min_val,
    const T max_val,
    const uint8_t b,
    ExceptionRule rule = ExceptionRule::None
) {
    GapComputation result{0, 0, 0, 0, 0, 0, 0, 0};
    const size_t n = v.size();
    if (n == 0) return result;

    using U = std::make_unsigned_t<T>;
    using WU = unsigned __int128;    // wide unsigned for safe arithmetic
    using WI = __int128;             // wide signed for gaps

    const auto is_exception = [&](size_t index, WI gap, size_t total_bits) -> bool {
        const int64_t hbits = std::max<int64_t>(0, static_cast<int64_t>(total_bits) - static_cast<int64_t>(b));
        switch (rule) {
            case ExceptionRule::None:
                return false;
            case ExceptionRule::BGEF: {
                if (index == 0) return true;
                const WU mag = gap < 0 ? static_cast<WU>(-gap) : static_cast<WU>(gap);
                return (mag + static_cast<WU>(2)) > static_cast<WU>(total_bits - b);
            }
            case ExceptionRule::UGEF: {
                if (index == 0) return true;
                if (gap < 0) return true;
                const WU g = static_cast<WU>(gap);
                return (g + static_cast<WU>(1)) > static_cast<WU>(hbits);
            }
        }
        return false;
    };

    const auto get_gap = [&](T current_raw, T previous_raw, uint8_t b) -> WI {
        // Compute high bits as unsigned after shifting to positive range
        const WI curr_signed = static_cast<WI>(current_raw) - static_cast<WI>(min_val);
        const WI prev_signed = static_cast<WI>(previous_raw) - static_cast<WI>(min_val);
        // Since min_val is min, curr_signed >= 0
        const WU qcurr = static_cast<WU>(curr_signed) >> b;
        const WU qprev = static_cast<WU>(prev_signed) >> b;
        return static_cast<WI>(qcurr) - static_cast<WI>(qprev);
    };


    // Compute range in 128-bit to avoid overflow and derive total_bits
    const WI min_w = static_cast<WI>(min_val);
    const WI max_w = static_cast<WI>(max_val);
    const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);
    const size_t total_bits = (range <= 1)
                              ? 1
                              : [] (WU r) {
                                    size_t bits = 0; WU x = r - 1;
                                    while (x > 0) { ++bits; x >>= 1; }
                                    return bits;
                                }(range);

    const WI first_gap = get_gap(v[0], min_val, b);
    result.sum_of_positive_gaps += static_cast<size_t>(first_gap);
    result.positive_gaps++;
    if (is_exception(0, first_gap, total_bits))
        result.positive_exceptions_count++;
    else
        result.sum_of_positive_gaps_without_exception += static_cast<size_t>(first_gap);

    for (size_t i = 1; i < n; ++i) {
        const WI gap = get_gap(v[i], v[i - 1], b);
        const bool exception = is_exception(i, gap, total_bits);
        if (gap >= 0) {
            const size_t g = static_cast<size_t>(gap);
            result.positive_gaps++;
            result.sum_of_positive_gaps += g;
            if (!exception) result.sum_of_positive_gaps_without_exception += g;
            if (exception)  result.positive_exceptions_count += 1;
        } else {
            const size_t g = static_cast<size_t>(-gap);
            result.negative_gaps++;
            if (!exception) result.sum_of_negative_gaps_without_exception += g;
            result.sum_of_negative_gaps += g;
            if (exception)  result.negative_exceptions_count += 1;
        }
    }








    return result;
}

// Replace the entire function starting from template<typename T> total_variation_of_shifted_vec_with_multiple_shifts to its closing brace with this corrected version

template<typename T>
std::vector<GapComputation>
total_variation_of_shifted_vec_with_multiple_shifts(
    const std::vector<T>& vec,
    const T min_val,
    const T max_val,
    const uint8_t min_b,
    const uint8_t max_b,
    ExceptionRule rule = ExceptionRule::None
) {
    if (min_b > max_b) return {};

    const size_t n = vec.size();
    const size_t num_shifts = max_b - min_b + 1;
    std::vector<GapComputation> results(num_shifts);

    if (n < 2) return results;

    // Precompute shifted values as unsigned to avoid signed issues
    using U = std::make_unsigned_t<T>;
    std::vector<U> shifted(n);
    for (size_t i = 0; i < n; ++i) {
        shifted[i] = static_cast<U>(vec[i] - min_val);
    }

    // Compute total_bits (used in both scalar and vector paths)
    using WU = unsigned __int128;
    using WI = __int128;
    const WI min_w = static_cast<WI>(min_val);
    const WI max_w = static_cast<WI>(max_val);
    const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);
    size_t total_bits = (range <= 1) ? 1 : [] (WU r) {
        size_t bits = 0; WU x = r - 1;
        while (x > 0) { ++bits; x >>= 1; }
        return bits;
    }(range);

    // Process in SIMD-friendly way: for each pair, compute for all b
#ifdef __AVX2__
    const size_t SIMD_LANES = 4; // AVX2: 256 bits / 64 bits = 4
    __m256i zero_vec = _mm256_setzero_si256();
    __m256i one_vec = _mm256_set1_epi64x(1);
    __m256i total_bits_vec = _mm256_set1_epi64x(total_bits);

    for (size_t b_group = min_b; b_group <= max_b; b_group += SIMD_LANES) {
        size_t group_size = std::min(SIMD_LANES, static_cast<size_t>(max_b - b_group + 1));
        
        // Build b_vec with actual b values (not 0 for unused lanes) to ensure correct indexing
        __m256i b_vec = _mm256_set_epi64x(
            (group_size > 3 ? b_group + 3 : b_group),
            (group_size > 2 ? b_group + 2 : b_group),
            (group_size > 1 ? b_group + 1 : b_group),
            b_group
        );

        // Accumulators
        __m256i sum_pos = zero_vec;
        __m256i sum_neg = zero_vec;
        __m256i sum_pos_no_exc = zero_vec;
        __m256i sum_neg_no_exc = zero_vec;
        __m256i pos_gaps = zero_vec;
        __m256i neg_gaps = zero_vec;
        __m256i pos_exc = zero_vec;
        __m256i neg_exc = zero_vec;

        // First gap (special case)
        __m256i curr_high = _mm256_srlv_epi64(_mm256_set1_epi64x(shifted[0]), b_vec);
        __m256i gap = curr_high; // First "prev" is 0
        __m256i is_neg = _mm256_cmpgt_epi64(zero_vec, gap);
        __m256i is_pos = _mm256_xor_si256(is_neg, _mm256_set1_epi64x(-1LL));
        // Custom abs
#ifdef __AVX512VL__
        __m256i mask = _mm256_srai_epi64(gap, 63);
#else
        __m256i mask = _mm256_cmpgt_epi64(zero_vec, gap);
#endif
        __m256i mag = _mm256_sub_epi64(_mm256_xor_si256(gap, mask), mask);
        __m256i is_exc = (rule == ExceptionRule::None) ? zero_vec : one_vec;
        __m256i pos_mask = is_pos;
        __m256i neg_mask = _mm256_xor_si256(pos_mask, _mm256_set1_epi64x(-1LL));
        sum_pos = _mm256_add_epi64(sum_pos, _mm256_and_si256(mag, pos_mask));
        sum_neg = _mm256_add_epi64(sum_neg, _mm256_and_si256(mag, neg_mask));
        pos_gaps = _mm256_add_epi64(pos_gaps, pos_mask);
        neg_gaps = _mm256_add_epi64(neg_gaps, neg_mask);
        __m256i no_exc_mask = _mm256_cmpeq_epi64(is_exc, zero_vec);
        sum_pos_no_exc = _mm256_add_epi64(sum_pos_no_exc, _mm256_and_si256(mag, _mm256_and_si256(pos_mask, no_exc_mask)));
        sum_neg_no_exc = _mm256_add_epi64(sum_neg_no_exc, _mm256_and_si256(mag, _mm256_and_si256(neg_mask, no_exc_mask)));
        pos_exc = _mm256_add_epi64(pos_exc, _mm256_and_si256(is_exc, pos_mask));
        neg_exc = _mm256_add_epi64(neg_exc, _mm256_and_si256(is_exc, neg_mask));

        // Main loop over pairs
        for (size_t i = 1; i < n; ++i) {
            __m256i prev_high = curr_high;
            curr_high = _mm256_srlv_epi64(_mm256_set1_epi64x(shifted[i]), b_vec);
            gap = _mm256_sub_epi64(curr_high, prev_high);
            is_neg = _mm256_cmpgt_epi64(zero_vec, gap);
            is_pos = _mm256_xor_si256(is_neg, _mm256_set1_epi64x(-1LL));
            // Custom abs
#ifdef __AVX512VL__
            __m256i mask = _mm256_srai_epi64(gap, 63);
#else
            __m256i mask = _mm256_cmpgt_epi64(zero_vec, gap);
#endif
            mag = _mm256_sub_epi64(_mm256_xor_si256(gap, mask), mask);
            is_exc = zero_vec;
            if (rule != ExceptionRule::None) {
                __m256i hbits = _mm256_sub_epi64(total_bits_vec, b_vec);
                if (rule == ExceptionRule::BGEF) {
                    __m256i threshold = _mm256_add_epi64(mag, _mm256_set1_epi64x(2));
                    is_exc = _mm256_cmpgt_epi64(threshold, hbits);
                } else if (rule == ExceptionRule::UGEF) {
                    __m256i threshold = _mm256_add_epi64(gap, one_vec);
                    __m256i is_large = _mm256_cmpgt_epi64(threshold, hbits);
                    is_exc = _mm256_or_si256(is_neg, is_large);
                }
            }
            pos_mask = is_pos;
            neg_mask = _mm256_xor_si256(pos_mask, _mm256_set1_epi64x(-1LL));
            sum_pos = _mm256_add_epi64(sum_pos, _mm256_and_si256(mag, pos_mask));
            sum_neg = _mm256_add_epi64(sum_neg, _mm256_and_si256(mag, neg_mask));
            pos_gaps = _mm256_add_epi64(pos_gaps, pos_mask);
            neg_gaps = _mm256_add_epi64(neg_gaps, neg_mask);
            no_exc_mask = _mm256_cmpeq_epi64(is_exc, zero_vec);
            sum_pos_no_exc = _mm256_add_epi64(sum_pos_no_exc, _mm256_and_si256(mag, _mm256_and_si256(pos_mask, no_exc_mask)));
            sum_neg_no_exc = _mm256_add_epi64(sum_neg_no_exc, _mm256_and_si256(mag, _mm256_and_si256(neg_mask, no_exc_mask)));
            pos_exc = _mm256_add_epi64(pos_exc, _mm256_and_si256(is_exc, pos_mask));
            neg_exc = _mm256_add_epi64(neg_exc, _mm256_and_si256(is_exc, neg_mask));
        }

        // Store results for this group
        alignas(32) uint64_t temp_results[4];
        // sum_of_positive_gaps
        _mm256_storeu_si256((__m256i*)temp_results, sum_pos);
        for (size_t j = 0; j < group_size; ++j) {
            results[b_group + j - min_b].sum_of_positive_gaps = temp_results[j];
        }
        // sum_of_negative_gaps
        _mm256_storeu_si256((__m256i*)temp_results, sum_neg);
        for (size_t j = 0; j < group_size; ++j) {
            results[b_group + j - min_b].sum_of_negative_gaps = temp_results[j];
        }
        // sum_of_positive_gaps_without_exception
        _mm256_storeu_si256((__m256i*)temp_results, sum_pos_no_exc);
        for (size_t j = 0; j < group_size; ++j) {
            results[b_group + j - min_b].sum_of_positive_gaps_without_exception = temp_results[j];
        }
        // sum_of_negative_gaps_without_exception
        _mm256_storeu_si256((__m256i*)temp_results, sum_neg_no_exc);
        for (size_t j = 0; j < group_size; ++j) {
            results[b_group + j - min_b].sum_of_negative_gaps_without_exception = temp_results[j];
        }
        // positive_gaps
        _mm256_storeu_si256((__m256i*)temp_results, pos_gaps);
        for (size_t j = 0; j < group_size; ++j) {
            results[b_group + j - min_b].positive_gaps = temp_results[j];
        }
        // negative_gaps
        _mm256_storeu_si256((__m256i*)temp_results, neg_gaps);
        for (size_t j = 0; j < group_size; ++j) {
            results[b_group + j - min_b].negative_gaps = temp_results[j];
        }
        // positive_exceptions_count
        _mm256_storeu_si256((__m256i*)temp_results, pos_exc);
        for (size_t j = 0; j < group_size; ++j) {
            results[b_group + j - min_b].positive_exceptions_count = temp_results[j];
        }
        // negative_exceptions_count
        _mm256_storeu_si256((__m256i*)temp_results, neg_exc);
        for (size_t j = 0; j < group_size; ++j) {
            results[b_group + j - min_b].negative_exceptions_count = temp_results[j];
        }
    }
#else
    // Fallback scalar implementation
    for (uint8_t b = min_b; b <= max_b; ++b) {
        results[b - min_b] = variation_of_shifted_vec(vec, min_val, max_val, b, rule);
    }
#endif
    return results;
}

// Refactored compute_all_gap_computations to use the SIMD version above
template<typename T>
std::vector<GapComputation> compute_all_gap_computations(
    const std::vector<T>& v,
    const T min_val,
    const T max_val,
    ExceptionRule rule,
    size_t total_bits
) {
    // Cap total_bits to what fits in uint8_t (max 255, but realistically <= 128 for __int128)
    // Note: Shifts of 64+ bits are defined (produce 0), so we allow the full range
    uint8_t max_b_capped = static_cast<uint8_t>(std::min(total_bits, static_cast<size_t>(255)));
    return total_variation_of_shifted_vec_with_multiple_shifts(v, min_val, max_val, 0, max_b_capped, rule);
}

#endif
