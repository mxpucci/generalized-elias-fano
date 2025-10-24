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

/**
 * For any b_i \in [min_b, max_b],
 * returns GapComputation(b_i) for b_i as in variation_of_shifted_vec.
 * Usually max_b  - min_b < 5
 */
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
    std::vector<GapComputation> results(max_b - min_b + 1);
    const size_t n = vec.size();
    if (n == 0) return results;

    for (size_t b = min_b; b <= max_b; ++b) {
        results[b - min_b] = variation_of_shifted_vec(
            vec,
            min_val,
            max_val,
            b,
            rule
        );
    }
    return results;
}

template<typename T>
std::vector<GapComputation> compute_all_gap_computations(
    const std::vector<T>& v,
    const T min_val,
    const T max_val,
    ExceptionRule rule,
    size_t total_bits
) {
    const size_t n = v.size();
    if (n == 0) return {};

    using U = std::make_unsigned_t<T>;
    using WU = unsigned __int128;
    using WI = __int128;

    std::vector<GapComputation> gcs(total_bits + 1);

    // Optimized exception checker: precompute thresholds to avoid repeated calculations
    auto is_exception_fast = [&](size_t index, int64_t gap, size_t b) -> bool {
        if (index == 0) return true;
        if (rule == ExceptionRule::None) return false;
        
        const int64_t hbits = static_cast<int64_t>(total_bits) - static_cast<int64_t>(b);
        if (hbits <= 0) return true;
        
        if (rule == ExceptionRule::UGEF) {
            // UGEF: negative gaps or gap >= h are exceptions
            return gap < 0 || gap >= hbits;
        } else {  // BGEF
            // BGEF: |gap| + 2 > h
            const int64_t mag = gap < 0 ? -gap : gap;
            return (mag + 2) > hbits;
        }
    };

    // Precompute shifted values (handle signed types correctly)
    // Use 64-bit when possible for performance
    if constexpr (sizeof(T) <= 8) {
        std::vector<uint64_t> shifted64(n);
        for (size_t i = 0; i < n; ++i) {
            const int64_t signed_diff = static_cast<int64_t>(v[i]) - static_cast<int64_t>(min_val);
            shifted64[i] = static_cast<uint64_t>(signed_diff);
        }

        // Handle first element for all b
        const uint64_t first_shifted = shifted64[0];
        for (size_t b = 0; b <= total_bits; ++b) {
            const int64_t first_gap = static_cast<int64_t>(first_shifted >> b);
            const size_t g = static_cast<size_t>(first_gap);
            gcs[b].positive_gaps++;
            gcs[b].sum_of_positive_gaps += g;
            const bool is_exc = is_exception_fast(0, first_gap, b);
            if (is_exc) {
                gcs[b].positive_exceptions_count++;
            } else {
                gcs[b].sum_of_positive_gaps_without_exception += g;
            }
        }

        // Process subsequent elements - compiler can auto-vectorize this better
        for (size_t i = 1; i < n; ++i) {
            const uint64_t curr = shifted64[i];
            const uint64_t prev = shifted64[i - 1];
            
            // Inner loop over b - this is where most time is spent
            // Unroll hint for compiler
            #pragma GCC unroll 16
            for (size_t b = 0; b <= total_bits; ++b) {
                const int64_t gap = static_cast<int64_t>(curr >> b) - static_cast<int64_t>(prev >> b);
                const bool is_exc = is_exception_fast(i, gap, b);
                
                // Branchless updates where possible
                const bool is_positive = gap >= 0;
                const size_t g = is_positive ? static_cast<size_t>(gap) : static_cast<size_t>(-gap);
                
                gcs[b].positive_gaps += is_positive;
                gcs[b].negative_gaps += !is_positive;
                gcs[b].sum_of_positive_gaps += is_positive ? g : 0;
                gcs[b].sum_of_negative_gaps += is_positive ? 0 : g;
                
                if (is_exc) {
                    gcs[b].positive_exceptions_count += is_positive;
                    gcs[b].negative_exceptions_count += !is_positive;
                } else {
                    gcs[b].sum_of_positive_gaps_without_exception += is_positive ? g : 0;
                    gcs[b].sum_of_negative_gaps_without_exception += is_positive ? 0 : g;
                }
            }
        }
    } else {
        // Fallback to 128-bit for larger types
        std::vector<WU> shifted(n);
        for (size_t i = 0; i < n; ++i) {
            const WI signed_diff = static_cast<WI>(v[i]) - static_cast<WI>(min_val);
            shifted[i] = static_cast<WU>(signed_diff);
        }

        const WU first_shifted = shifted[0];
        for (size_t b = 0; b <= total_bits; ++b) {
            const WI first_gap = static_cast<WI>(first_shifted >> b);
            const size_t g = static_cast<size_t>(first_gap);
            gcs[b].positive_gaps++;
            gcs[b].sum_of_positive_gaps += g;
            const bool is_exc = is_exception_fast(0, first_gap, b);
            if (is_exc) {
                gcs[b].positive_exceptions_count++;
            } else {
                gcs[b].sum_of_positive_gaps_without_exception += g;
            }
        }

        for (size_t i = 1; i < n; ++i) {
            const WU curr = shifted[i];
            const WU prev = shifted[i - 1];
            for (size_t b = 0; b <= total_bits; ++b) {
                const WI gap = static_cast<WI>(curr >> b) - static_cast<WI>(prev >> b);
                const bool is_exc = is_exception_fast(i, gap, b);
                if (gap >= 0) {
                    const size_t g = static_cast<size_t>(gap);
                    gcs[b].positive_gaps++;
                    gcs[b].sum_of_positive_gaps += g;
                    if (!is_exc) gcs[b].sum_of_positive_gaps_without_exception += g;
                    else gcs[b].positive_exceptions_count++;
                } else {
                    const size_t g = static_cast<size_t>(-gap);
                    gcs[b].negative_gaps++;
                    gcs[b].sum_of_negative_gaps += g;
                    if (!is_exc) gcs[b].sum_of_negative_gaps_without_exception += g;
                    else gcs[b].negative_exceptions_count++;
                }
            }
        }
    }

    return gcs;
}

#endif
