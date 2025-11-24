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

#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
#include <immintrin.h>
#endif

using namespace gef;

enum class ExceptionRule : uint8_t {
    None  = 0,  // Rule #0  (B_GEF_STAR): never an exception
    BGEF  = 1,  // Rule #1  (B_GEF):      i==0 || abs(gap)+2 > total_bits - b
    UGEF  = 2   // Rule #2  (U_GEF):      i==0 || abs(gap)+1 > total_bits - b (Symmetric magnitude check)
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
    size_t neg_gaps = 0;
    size_t zero_gaps = 0;
    size_t sum_pos = 0;
    size_t sum_neg = 0;

    // first gap is always non-negative since min_val <= vec[0]
    {
        sum_pos += (static_cast<uint64_t>(vec[0]) - static_cast<uint64_t>(min_val));
        ++pos_gaps;             // count the initial positive gap
    }

    // Optimized scalar loop - compiler can auto-vectorize this with -O3 -march=native
    // Manual SIMD adds complexity without much benefit for modern compilers
    for (size_t i = 1; i < n; ++i) {
        // Use signed arithmetic for correct handling of negative values
        // Using __int128 to prevent overflow when calculating diff of uint64_t values that are far apart
        // (e.g. 0 and 2^64-1) which would wrap in int64_t.
        using WI = __int128;
        const WI diff = static_cast<WI>(vec[i]) - static_cast<WI>(vec[i - 1]);

        // Branchless computation - compiler can vectorize this
        const bool is_positive = diff >= 0;
        const uint64_t mag = is_positive ? static_cast<uint64_t>(diff) : static_cast<uint64_t>(-diff);

        if (is_positive) {
            ++pos_gaps;
            if (diff == 0) {
                ++zero_gaps;
            }
        } else {
            ++neg_gaps;
        }
        sum_pos += is_positive ? mag : 0;
        sum_neg += is_positive ? 0 : mag;
    }

    result.positive_gaps = pos_gaps;
    result.negative_gaps = neg_gaps;
    result.zero_gaps = zero_gaps;
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
    GapComputation result{};
    const size_t n = v.size();
    if (n == 0) return result;

    size_t zero_gap_count = 0;

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
                // Symmetric check for U_GEF as well to allow inferring reversed stats
                const WU mag = gap < 0 ? static_cast<WU>(-gap) : static_cast<WU>(gap);
                return (mag + static_cast<WU>(1)) > static_cast<WU>(hbits);
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
            if (gap == 0) {
                ++zero_gap_count;
            }
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

    result.zero_gaps = zero_gap_count;
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

    // Compute total_bits (used in both scalar and vector paths)
    using U = std::make_unsigned_t<T>;
    using WU = unsigned __int128;
    using WI = __int128;
    const WI min_w = static_cast<WI>(min_val);
    const WI max_w = static_cast<WI>(max_val);
    const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);
    const size_t total_bits = (range <= 1) ? 1 : [] (WU r) {
        size_t bits = 0; WU x = r - 1;
        while (x > 0) { ++bits; x >>= 1; }
        return bits;
    }(range);

    // Process in SIMD-friendly way: for each pair, compute for all b
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
    const size_t SIMD_LANES = 4; // AVX2: 256 bits / 64 bits = 4
    __m256i zero_vec = _mm256_setzero_si256();
    __m256i one_vec = _mm256_set1_epi64x(1);
    __m256i total_bits_vec = _mm256_set1_epi64x(total_bits);

    auto ugt = [&](__m256i a, __m256i b) -> __m256i {
        __m256i signed_gt = _mm256_cmpgt_epi64(a, b);
        __m256i a_neg = _mm256_cmpgt_epi64(zero_vec, a);
        __m256i b_neg = _mm256_cmpgt_epi64(zero_vec, b);
        __m256i xor_signs = _mm256_xor_si256(a_neg, b_neg);
        return _mm256_xor_si256(signed_gt, xor_signs);
    };

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
        __m256i zero_gaps = zero_vec;
        __m256i pos_exc = zero_vec;
        __m256i neg_exc = zero_vec;

        // First gap (special case)
        WI val_0 = static_cast<WI>(vec[0]) - static_cast<WI>(min_val);
        __m256i curr_high = _mm256_srlv_epi64(_mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(val_0))), b_vec);
        __m256i prev_high = zero_vec;
        __m256i signed_less = _mm256_cmpgt_epi64(prev_high, curr_high);
        __m256i curr_neg = _mm256_cmpgt_epi64(zero_vec, curr_high);
        __m256i prev_neg = _mm256_cmpgt_epi64(zero_vec, prev_high);
        __m256i xor_signs = _mm256_xor_si256(curr_neg, prev_neg);
        __m256i is_neg = _mm256_xor_si256(signed_less, xor_signs);
        __m256i is_pos = _mm256_xor_si256(is_neg, _mm256_set1_epi64x(-1LL));
        __m256i pos_diff = _mm256_sub_epi64(curr_high, prev_high);
        __m256i neg_diff = _mm256_sub_epi64(prev_high, curr_high);
        __m256i mag = _mm256_blendv_epi8(pos_diff, neg_diff, is_neg);
        __m256i is_exc = (rule == ExceptionRule::None) ? zero_vec : _mm256_set1_epi64x(-1LL);
        __m256i pos_mask = is_pos;
        __m256i neg_mask = _mm256_xor_si256(pos_mask, _mm256_set1_epi64x(-1LL));
        sum_pos = _mm256_add_epi64(sum_pos, _mm256_and_si256(mag, pos_mask));
        sum_neg = _mm256_add_epi64(sum_neg, _mm256_and_si256(mag, neg_mask));
        pos_gaps = _mm256_add_epi64(pos_gaps, _mm256_srli_epi64(pos_mask, 63));
        neg_gaps = _mm256_add_epi64(neg_gaps, _mm256_srli_epi64(neg_mask, 63));
        __m256i no_exc_mask = _mm256_cmpeq_epi64(is_exc, zero_vec);
        sum_pos_no_exc = _mm256_add_epi64(sum_pos_no_exc, _mm256_and_si256(mag, _mm256_and_si256(pos_mask, no_exc_mask)));
        sum_neg_no_exc = _mm256_add_epi64(sum_neg_no_exc, _mm256_and_si256(mag, _mm256_and_si256(neg_mask, no_exc_mask)));
        pos_exc = _mm256_add_epi64(pos_exc, _mm256_srli_epi64(_mm256_and_si256(is_exc, pos_mask), 63));
        neg_exc = _mm256_add_epi64(neg_exc, _mm256_srli_epi64(_mm256_and_si256(is_exc, neg_mask), 63));
        
        // Main loop over pairs
        for (size_t i = 1; i < n; ++i) {
            prev_high = curr_high;
            WI val_i = static_cast<WI>(vec[i]) - static_cast<WI>(min_val);
            curr_high = _mm256_srlv_epi64(_mm256_set1_epi64x(static_cast<long long>(static_cast<uint64_t>(val_i))), b_vec);
            __m256i eq_mask = _mm256_cmpeq_epi64(curr_high, prev_high);
            zero_gaps = _mm256_add_epi64(zero_gaps, _mm256_srli_epi64(eq_mask, 63));
            signed_less = _mm256_cmpgt_epi64(prev_high, curr_high);
            curr_neg = _mm256_cmpgt_epi64(zero_vec, curr_high);
            prev_neg = _mm256_cmpgt_epi64(zero_vec, prev_high);
            xor_signs = _mm256_xor_si256(curr_neg, prev_neg);
            is_neg = _mm256_xor_si256(signed_less, xor_signs);
            is_pos = _mm256_xor_si256(is_neg, _mm256_set1_epi64x(-1LL));
            pos_diff = _mm256_sub_epi64(curr_high, prev_high);
            neg_diff = _mm256_sub_epi64(prev_high, curr_high);
            mag = _mm256_blendv_epi8(pos_diff, neg_diff, is_neg);
            is_exc = zero_vec;
            
            if (rule != ExceptionRule::None) {
                __m256i hbits = _mm256_sub_epi64(total_bits_vec, b_vec);
                __m256i gt_zero = _mm256_cmpgt_epi64(hbits, zero_vec);
                hbits = _mm256_blendv_epi8(zero_vec, hbits, gt_zero);
                
                if (rule == ExceptionRule::BGEF) {
                    __m256i threshold = _mm256_add_epi64(mag, _mm256_set1_epi64x(2));
                    is_exc = ugt(threshold, hbits);
                } else if (rule == ExceptionRule::UGEF) {
                    __m256i threshold = _mm256_add_epi64(mag, one_vec);
                    is_exc = ugt(threshold, hbits);
                }
            }
            
            pos_mask = is_pos;
            neg_mask = _mm256_xor_si256(pos_mask, _mm256_set1_epi64x(-1LL));
            sum_pos = _mm256_add_epi64(sum_pos, _mm256_and_si256(mag, pos_mask));
            sum_neg = _mm256_add_epi64(sum_neg, _mm256_and_si256(mag, neg_mask));
            pos_gaps = _mm256_add_epi64(pos_gaps, _mm256_srli_epi64(pos_mask, 63));
            neg_gaps = _mm256_add_epi64(neg_gaps, _mm256_srli_epi64(neg_mask, 63));
            no_exc_mask = _mm256_cmpeq_epi64(is_exc, zero_vec);
            sum_pos_no_exc = _mm256_add_epi64(sum_pos_no_exc, _mm256_and_si256(mag, _mm256_and_si256(pos_mask, no_exc_mask)));
            sum_neg_no_exc = _mm256_add_epi64(sum_neg_no_exc, _mm256_and_si256(mag, _mm256_and_si256(neg_mask, no_exc_mask)));
            pos_exc = _mm256_add_epi64(pos_exc, _mm256_srli_epi64(_mm256_and_si256(is_exc, pos_mask), 63));
            neg_exc = _mm256_add_epi64(neg_exc, _mm256_srli_epi64(_mm256_and_si256(is_exc, neg_mask), 63));
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
        // zero_gaps
        _mm256_storeu_si256((__m256i*)temp_results, zero_gaps);
        for (size_t j = 0; j < group_size; ++j) {
            results[b_group + j - min_b].zero_gaps = temp_results[j];
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
