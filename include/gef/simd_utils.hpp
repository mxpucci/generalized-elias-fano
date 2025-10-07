//
// Created by Michelangelo Pucci on 07/10/25.
//

#ifndef SIMD_UTILS_HPP
#define SIMD_UTILS_HPP

#include "IGEF.hpp"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <bit>

#ifdef __AVX2__
#include <immintrin.h>
#endif

using namespace gef;

/*
 * positive_gaps = # i s.t. s[i] >= s[i - 1]
 * negative_gaps = n - positive_gaps
 * positive_exceptions_count = negative_exceptions_count = 0
 * sum_of_positive_gaps = sum_of_positive_gaps_without_exception = \sum { s_i - s_{i - 1} : s[i] >= s[i - 1]}
 * sum_of_negative_gaps = sum_of_negative_gaps_without_exception = \sum { s_{i - 1} - s_i : s[i] < s[i - 1]}
 */
template<typename T>
GapComputation simd_optimized_variation_of_original_vec(
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

    auto scalar_compute = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            T curr = vec[i];
            T prev = vec[i - 1];
            if (curr >= prev) {
                ++pos_gaps;
                uint64_t a = static_cast<uint64_t>(curr);
                uint64_t b = static_cast<uint64_t>(prev);
                sum_pos += a - b;
            } else {
                uint64_t a = static_cast<uint64_t>(curr);
                uint64_t b = static_cast<uint64_t>(prev);
                sum_neg += b - a;
            }
        }
    };

#ifdef __AVX2__
    size_t num_simd = (n - 1) / 4;
    size_t end_simd_gap = num_simd * 4;
    const T* data = vec.data();

    for (size_t k = 0; k < num_simd; ++k) {
        size_t idx = k * 4;
        __m256i prev = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + idx));
        __m256i curr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + idx + 1));

        __m256i mask_ge;
        if constexpr (std::is_signed_v<T>) {
            __m256i gt = _mm256_cmpgt_epi64(curr, prev);
            __m256i eq = _mm256_cmpeq_epi64(curr, prev);
            mask_ge = _mm256_or_si256(gt, eq);
        } else {
            __m256i flip = _mm256_set1_epi64x(1LL << 63);
            __m256i curr_flip = _mm256_xor_si256(curr, flip);
            __m256i prev_flip = _mm256_xor_si256(prev, flip);
            __m256i gt = _mm256_cmpgt_epi64(curr_flip, prev_flip);
            __m256i eq = _mm256_cmpeq_epi64(curr_flip, prev_flip);
            mask_ge = _mm256_or_si256(gt, eq);
        }

        __m256i diff = _mm256_sub_epi64(curr, prev);
        __m256i neg_diff = _mm256_sub_epi64(prev, curr);
        __m256i pos_contrib = _mm256_and_si256(mask_ge, diff);
        __m256i neg_contrib = _mm256_andnot_si256(mask_ge, neg_diff);

        sum_pos += static_cast<size_t>(_mm256_extract_epi64(pos_contrib, 0)) +
                   static_cast<size_t>(_mm256_extract_epi64(pos_contrib, 1)) +
                   static_cast<size_t>(_mm256_extract_epi64(pos_contrib, 2)) +
                   static_cast<size_t>(_mm256_extract_epi64(pos_contrib, 3));

        sum_neg += static_cast<size_t>(_mm256_extract_epi64(neg_contrib, 0)) +
                   static_cast<size_t>(_mm256_extract_epi64(neg_contrib, 1)) +
                   static_cast<size_t>(_mm256_extract_epi64(neg_contrib, 2)) +
                   static_cast<size_t>(_mm256_extract_epi64(neg_contrib, 3));

        int movemask = _mm256_movemask_pd(_mm256_castsi256_pd(mask_ge));
        size_t pos_this = __builtin_popcount(movemask);
        pos_gaps += pos_this;
    }

    // Remaining scalar
    scalar_compute(end_simd_gap + 1, n);
#else
    scalar_compute(1, n);
#endif

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

/*
 * positive_gaps = # i s.t. floor((s[i] - *min_it) / pow(2, b)) >= floor((s[i - 1] - *min_it) / pow(2, b))
 * negative_gaps = n - positive_gaps
 * sum_of_positive_gaps = \sum \max{floor((s[i] - *min_it) / pow(2, b)) - floor((s[i - 1] - *min_it) / pow(2, b)), 0}
 * sum_of_negative_gaps = \sum - \min{floor((s[i] - *min_it) / pow(2, b)) - floor((s[i - 1] - *min_it) / pow(2, b)), 0}
 * positive_exceptions_count = # i s.t. \max{floor((s[i] - *min_it) / pow(2, b)) - floor((s[i - 1] - *min_it) / pow(2, b)), 0} > ceil(log_2(*max_it - *min_it + 1)) - b
 * negative_exceptions_count = # i s.t. -\min{floor((s[i] - *min_it) / pow(2, b)) - floor((s[i - 1] - *min_it) / pow(2, b)), 0} > ceil(log_2(*max_it - *min_it + 1)) - b
 */
template<typename T>
GapComputation simd_optimized_variation_of_shifted_vec(
    const std::vector<T>& v,
    const T min_val,
    const T max_val,
    const uint8_t b
) {
    GapComputation result{};
    size_t n = v.size();
    if (n < 2) return result;


    const uint64_t min_u = static_cast<uint64_t>(min_val);
    const uint64_t max_u = static_cast<uint64_t>(max_val);
    const uint64_t range = max_u - min_u + 1;

    size_t ceil_lg = 0;
    if (range > 1) {
        uint64_t t = range - 1;
        ceil_lg = 64 - __builtin_clzll(t);
    }
    size_t threshold = ceil_lg > b ? ceil_lg - b : 0;

    size_t pos_gaps = 0;
    size_t sum_pos = 0;
    size_t sum_neg = 0;
    size_t sum_pos_without = 0;
    size_t sum_neg_without = 0;
    size_t pos_exc = 0;
    size_t neg_exc = 0;

    // --- first gap: from 0 to val0 ---
    {
        const uint64_t curr_raw0 = static_cast<uint64_t>(v[0]);
        const uint64_t val0 = (curr_raw0 - min_u) >> b;
        ++pos_gaps;            // val0 >= 0 always
        sum_pos += val0;
        if (val0 > threshold) ++pos_exc;
        else                  sum_pos_without += val0;
    }

    auto scalar_compute = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            uint64_t prev_raw = static_cast<uint64_t>(v[i - 1]);
            uint64_t curr_raw = static_cast<uint64_t>(v[i]);
            uint64_t val_prev = (prev_raw - min_u) >> b;
            uint64_t val_curr = (curr_raw - min_u) >> b;
            if (val_curr >= val_prev) {
                ++pos_gaps;
                uint64_t gap = val_curr - val_prev;
                sum_pos += gap;
                if (gap > threshold) {
                    ++pos_exc;
                } else {
                    sum_pos_without += gap;
                }
            } else {
                uint64_t gap = val_prev - val_curr;
                sum_neg += gap;
                if (gap > threshold) {
                    ++neg_exc;
                } else {
                    sum_neg_without += gap;
                }
            }
        }
    };

#ifdef __AVX2__
    size_t num_simd = (n - 1) / 4;
    size_t end_simd_gap = num_simd * 4;
    const T* data = v.data();

    for (size_t k = 0; k < num_simd; ++k) {
        size_t idx = k * 4;
        __m256i prev = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + idx));
        __m256i curr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + idx + 1));

        __m256i min_vec = _mm256_set1_epi64x(static_cast<long long>(min_val));

        __m256i val_prev = _mm256_srli_epi64(_mm256_sub_epi64(prev, min_vec), b);
        __m256i val_curr = _mm256_srli_epi64(_mm256_sub_epi64(curr, min_vec), b);

        __m256i mask_ge;
        if constexpr (std::is_signed_v<T>) {
            __m256i gt = _mm256_cmpgt_epi64(val_curr, val_prev);
            __m256i eq = _mm256_cmpeq_epi64(val_curr, val_prev);
            mask_ge = _mm256_or_si256(gt, eq);
        } else {
            __m256i flip = _mm256_set1_epi64x(1LL << 63);
            __m256i curr_flip = _mm256_xor_si256(val_curr, flip);
            __m256i prev_flip = _mm256_xor_si256(val_prev, flip);
            __m256i gt = _mm256_cmpgt_epi64(curr_flip, prev_flip);
            __m256i eq = _mm256_cmpeq_epi64(curr_flip, prev_flip);
            mask_ge = _mm256_or_si256(gt, eq);
        }

        __m256i diff = _mm256_sub_epi64(val_curr, val_prev);
        __m256i neg_diff = _mm256_sub_epi64(val_prev, val_curr);
        __m256i pos_contrib = _mm256_and_si256(mask_ge, diff);
        __m256i neg_contrib = _mm256_andnot_si256(mask_ge, neg_diff);

        for (int lane = 0; lane < 4; ++lane) {
            uint64_t pos_gap = static_cast<uint64_t>(_mm256_extract_epi64(pos_contrib, lane));
            uint64_t neg_gap = static_cast<uint64_t>(_mm256_extract_epi64(neg_contrib, lane));
            sum_pos += pos_gap;
            sum_neg += neg_gap;
            if (pos_gap > threshold) {
                ++pos_exc;
            } else {
                sum_pos_without += pos_gap;
            }
            if (neg_gap > threshold) {
                ++neg_exc;
            } else {
                sum_neg_without += neg_gap;
            }
        }

        int movemask = _mm256_movemask_pd(_mm256_castsi256_pd(mask_ge));
        size_t pos_this = __builtin_popcount(movemask);
        pos_gaps += pos_this;
    }

    // Remaining scalar
    scalar_compute(end_simd_gap + 1, n);
#else
    scalar_compute(1, n);
#endif

    result.positive_gaps = pos_gaps;
    result.negative_gaps = (n - 1) - pos_gaps;
    result.positive_exceptions_count = pos_exc;
    result.negative_exceptions_count = neg_exc;
    result.sum_of_positive_gaps = sum_pos;
    result.sum_of_negative_gaps = sum_neg;
    result.sum_of_positive_gaps_without_exception = sum_pos_without;
    result.sum_of_negative_gaps_without_exception = sum_neg_without;
    return result;
}



/**
 * For any b_i \in [min_b, max_b],
 * returns GapComputation(b_i) for b_i as in simd_optimized_variation_of_shifted_vec.
 * Usually max_b  - min_b < 5
 */
template<typename T>
std::vector<GapComputation> simd_optimized_total_variation_of_shifted_vec_with_multiple_shifts
(const std::vector<T>& vec,
    const T min_val,
    const T max_val,
    const uint8_t min_b,
    const uint8_t max_b) {
    std::vector<GapComputation> results;
    size_t n = vec.size();
    uint8_t num_b_ = max_b - min_b + 1;
    if (num_b_ == 0 || n < 2) {
        results.resize(num_b_);
        return results;
    }


    uint64_t min_u = static_cast<uint64_t>(min_val);
    uint64_t max_u = static_cast<uint64_t>(max_val);
    uint64_t range = max_u - min_u + 1;

    size_t ceil_lg = 0;
    if (range > 1) {
        uint64_t t = range - 1;
        ceil_lg = 64 - __builtin_clzll(t);
    }

    std::vector<size_t> thresholds(num_b_);
    for (uint8_t bb = 0; bb < num_b_; ++bb) {
        uint8_t b = min_b + bb;
        thresholds[bb] = (ceil_lg > b) ? ceil_lg - b : 0;
    }

    std::vector<size_t> pos_gaps(num_b_, 0);
    std::vector<size_t> pos_exc(num_b_, 0);
    std::vector<size_t> neg_exc(num_b_, 0);
    std::vector<size_t> sum_pos(num_b_, 0);
    std::vector<size_t> sum_neg(num_b_, 0);
    std::vector<size_t> sum_pos_without(num_b_, 0);
    std::vector<size_t> sum_neg_without(num_b_, 0);

    // --- first gap for each b: from 0 to ((vec[0]-min)>>b) ---
    {
        const uint64_t raw0 = static_cast<uint64_t>(vec[0]);
        const uint64_t shifted0 = raw0 - min_u;  // non-negative
        for (uint8_t bb = 0; bb < num_b_; ++bb) {
            const uint8_t b = min_b + bb;
            const uint64_t val0 = shifted0 >> b;
            ++pos_gaps[bb];
            sum_pos[bb] += val0;
            if (val0 > thresholds[bb]) ++pos_exc[bb];
            else                       sum_pos_without[bb] += val0;
        }
    }

    auto scalar_compute = [&](size_t start, size_t end) {
        for (size_t i = start; i < n; ++i) {
            uint64_t prev_raw = static_cast<uint64_t>(vec[i - 1]);
            uint64_t curr_raw = static_cast<uint64_t>(vec[i]);
            uint64_t shifted_prev = prev_raw - min_u;
            uint64_t shifted_curr = curr_raw - min_u;
            for (uint8_t bb = 0; bb < num_b_; ++bb) {
                uint8_t b = min_b + bb;
                uint64_t val_prev = shifted_prev >> b;
                uint64_t val_curr = shifted_curr >> b;
                size_t idx = bb;
                size_t thresh = thresholds[idx];
                if (val_curr >= val_prev) {
                    ++pos_gaps[idx];
                    uint64_t gap = val_curr - val_prev;
                    sum_pos[idx] += gap;
                    if (gap > thresh) {
                        ++pos_exc[idx];
                    } else {
                        sum_pos_without[idx] += gap;
                    }
                } else {
                    uint64_t gap = val_prev - val_curr;
                    sum_neg[idx] += gap;
                    if (gap > thresh) {
                        ++neg_exc[idx];
                    } else {
                        sum_neg_without[idx] += gap;
                    }
                }
            }
        }
    };

    size_t start_remaining = 1;
#if defined(__AVX2__)
    size_t num_gaps = n - 1;
    size_t num_simd = num_gaps / 4;
    size_t end_simd_gap = num_simd * 4;
    start_remaining = end_simd_gap + 1;
    const T* data = vec.data();

    for (size_t k = 0; k < num_simd; ++k) {
        size_t idx = k * 4;
        __m256i prev = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + idx));
        __m256i curr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + idx + 1));

        __m256i min_vec = _mm256_set1_epi64x(static_cast<long long>(min_val));

        __m256i shifted_prev = _mm256_sub_epi64(prev, min_vec);
        __m256i shifted_curr = _mm256_sub_epi64(curr, min_vec);

        for (uint8_t bb = 0; bb < num_b_; ++bb) {
            uint8_t b = min_b + bb;
            __m256i val_prev = _mm256_srli_epi64(shifted_prev, b);
            __m256i val_curr = _mm256_srli_epi64(shifted_curr, b);

            __m256i mask_ge;
            if constexpr (std::is_signed_v<T>) {
                __m256i gt = _mm256_cmpgt_epi64(val_curr, val_prev);
                __m256i eq = _mm256_cmpeq_epi64(val_curr, val_prev);
                mask_ge = _mm256_or_si256(gt, eq);
            } else {
                __m256i flip = _mm256_set1_epi64x(1LL << 63);
                __m256i curr_flip = _mm256_xor_si256(val_curr, flip);
                __m256i prev_flip = _mm256_xor_si256(val_prev, flip);
                __m256i gt = _mm256_cmpgt_epi64(curr_flip, prev_flip);
                __m256i eq = _mm256_cmpeq_epi64(curr_flip, prev_flip);
                mask_ge = _mm256_or_si256(gt, eq);
            }

            __m256i diff = _mm256_sub_epi64(val_curr, val_prev);
            __m256i neg_diff = _mm256_sub_epi64(val_prev, val_curr);
            __m256i pos_contrib = _mm256_and_si256(mask_ge, diff);
            __m256i neg_contrib = _mm256_andnot_si256(mask_ge, neg_diff);

            size_t idx_b = bb;
            size_t thresh = thresholds[idx_b];

            for (int lane = 0; lane < 4; ++lane) {
                uint64_t pos_gap = static_cast<uint64_t>(_mm256_extract_epi64(pos_contrib, lane));
                uint64_t neg_gap = static_cast<uint64_t>(_mm256_extract_epi64(neg_contrib, lane));
                sum_pos[idx_b] += pos_gap;
                sum_neg[idx_b] += neg_gap;
                if (pos_gap > thresh) {
                    ++pos_exc[idx_b];
                } else {
                    sum_pos_without[idx_b] += pos_gap;
                }
                if (neg_gap > thresh) {
                    ++neg_exc[idx_b];
                } else {
                    sum_neg_without[idx_b] += neg_gap;
                }
            }

            int movemask = _mm256_movemask_pd(_mm256_castsi256_pd(mask_ge));
            size_t pos_this = __builtin_popcount(movemask);
            pos_gaps[idx_b] += pos_this;
        }
    }
#endif

    // Remaining scalar
    scalar_compute(start_remaining, n);

    results.resize(num_b_);
    for (uint8_t bb = 0; bb < num_b_; ++bb) {
        size_t idx = bb;
        results[idx].positive_gaps = pos_gaps[idx];
        results[idx].negative_gaps = (n - 1) - pos_gaps[idx];
        results[idx].positive_exceptions_count = pos_exc[idx];
        results[idx].negative_exceptions_count = neg_exc[idx];
        results[idx].sum_of_positive_gaps = sum_pos[idx];
        results[idx].sum_of_negative_gaps = sum_neg[idx];
        results[idx].sum_of_positive_gaps_without_exception = sum_pos_without[idx];
        results[idx].sum_of_negative_gaps_without_exception = sum_neg_without[idx];
    }
    return results;
}

#endif //SIMD_UTILS_HPP
