//
// Created by Michelangelo Pucci on 03/08/25.
//

#ifndef B_GEF_NO_RLE_NO_RLE_HPP
#define B_GEF_NO_RLE_NO_RLE_HPP

#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits>
#include "IGEF.hpp"
#include "RLE_GEF.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"

#if defined(__AVX2__) || defined(__SSE4_2__)
  #include <immintrin.h>
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace gef {
    template<typename T>
    class B_GEF_NO_RLE : public IGEF<T> {
    private:
        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that highPart(i) >= highPart(i - 1)
         */
        std::unique_ptr<IBitVector> G_plus;


        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that highPart(i - 1) >= highPart(i)
         */
        std::unique_ptr<IBitVector> G_minus;

        // low parts
        sdsl::int_vector<> L;

        // The split point that rules which bits are stored in H and in L
        uint8_t b;
        uint8_t h;

        /**
         * The minimum of the encoded sequence, so that we store the shifted sequence
         * that falls in the range [0, max S - base]
         * This tricks may boost compression and allows us to implicitly store negative numbers
         */
        T base;

        static size_t evaluate_space(const std::vector<T> &S, const T &base, const uint8_t total_bits, uint8_t b,
                                     const std::shared_ptr<IBitVectorFactory> &factory) {
            const size_t N = S.size();
            if (N == 0) {
                return sizeof(T) + sizeof(uint8_t) * 2; // Overhead for base, h, b
            }

            if (b >= total_bits) {
                return N * total_bits;
            }

            const uint8_t h = static_cast<uint8_t>(total_bits - b);

            size_t g_plus_unary_bits = 0;
            size_t g_minus_unary_bits = 0;

            // lastHighBits starts at 0 (as in original)
            uint64_t lastHighBits = 0;

            // SIMD-accelerated adjacent-diff accumulation in 64-bit lanes.
            size_t i = 0;

#if defined(__AVX2__)
    // Process 4 elements per iteration
    for (; i + 4 <= N; i += 4) {
        // Compute highPart for the block (preserve exact semantics)
        const uint64_t hb0 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 0] - base), total_bits, h));
        const uint64_t hb1 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 1] - base), total_bits, h));
        const uint64_t hb2 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 2] - base), total_bits, h));
        const uint64_t hb3 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 3] - base), total_bits, h));

        // vcurr = [hb0, hb1, hb2, hb3]
        alignas(32) uint64_t tmpCurr[4] = { hb0, hb1, hb2, hb3 };
        const __m256i vcurr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpCurr));

        // vprev = [last, hb0, hb1, hb2]
        const __m128i prev_lo = _mm_set_epi64x(static_cast<long long>(hb0), static_cast<long long>(lastHighBits));
        const __m128i prev_hi = _mm_set_epi64x(static_cast<long long>(hb2), static_cast<long long>(hb1));
        const __m256i vprev = _mm256_set_m128i(prev_hi, prev_lo);

        const __m256i vdiff = _mm256_sub_epi64(vcurr, vprev);

        const __m256i vzero = _mm256_setzero_si256();
        const __m256i maskPos = _mm256_cmpgt_epi64(vdiff, vzero);
        const __m256i maskNeg = _mm256_cmpgt_epi64(vzero, vdiff);

        const __m256i vpos = _mm256_blendv_epi8(vzero, vdiff, maskPos);
        const __m256i vneg = _mm256_blendv_epi8(vzero, _mm256_sub_epi64(vzero, vdiff), maskNeg);

        alignas(32) int64_t pos[4];
        alignas(32) int64_t neg[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pos), vpos);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(neg), vneg);

        g_plus_unary_bits  += static_cast<size_t>(pos[0]) + static_cast<size_t>(pos[1])
                            + static_cast<size_t>(pos[2]) + static_cast<size_t>(pos[3]);
        g_minus_unary_bits += static_cast<size_t>(neg[0]) + static_cast<size_t>(neg[1])
                            + static_cast<size_t>(neg[2]) + static_cast<size_t>(neg[3]);

        lastHighBits = hb3;
    }
#elif defined(__SSE4_2__)
    // Process 2 elements per iteration
    for (; i + 2 <= N; i += 2) {
        const uint64_t hb0 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 0] - base), total_bits, h));
        const uint64_t hb1 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 1] - base), total_bits, h));

        const __m128i vcurr = _mm_set_epi64x(static_cast<long long>(hb1), static_cast<long long>(hb0));
        const __m128i vprev = _mm_set_epi64x(static_cast<long long>(hb0), static_cast<long long>(lastHighBits));

        const __m128i vdiff = _mm_sub_epi64(vcurr, vprev);

        const __m128i vzero = _mm_setzero_si128();
        const __m128i maskPos = _mm_cmpgt_epi64(vdiff, vzero);
        const __m128i maskNeg = _mm_cmpgt_epi64(vzero, vdiff);

        const __m128i vpos = _mm_blendv_epi8(vzero, vdiff, maskPos);
        const __m128i vneg = _mm_blendv_epi8(vzero, _mm_sub_epi64(vzero, vdiff), maskNeg);

        alignas(16) int64_t pos[2];
        alignas(16) int64_t neg[2];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(pos), vpos);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(neg), vneg);

        g_plus_unary_bits  += static_cast<size_t>(pos[0]) + static_cast<size_t>(pos[1]);
        g_minus_unary_bits += static_cast<size_t>(neg[0]) + static_cast<size_t>(neg[1]);

        lastHighBits = hb1;
    }
#elif defined(__aarch64__) && defined(__ARM_NEON)
            // Process 2 elements per iteration
            for (; i + 2 <= N; i += 2) {
                const int64_t hb0 = static_cast<int64_t>(highPart(static_cast<T>(S[i + 0] - base), total_bits, h));
                const int64_t hb1 = static_cast<int64_t>(highPart(static_cast<T>(S[i + 1] - base), total_bits, h));

                int64x2_t vcurr = vsetq_lane_s64(hb0, vdupq_n_s64(0), 0);
                vcurr = vsetq_lane_s64(hb1, vcurr, 1);

                int64x2_t vprev = vsetq_lane_s64(static_cast<int64_t>(lastHighBits), vdupq_n_s64(0), 0);
                vprev = vsetq_lane_s64(hb0, vprev, 1);

                int64x2_t vdiff = vsubq_s64(vcurr, vprev);

                const int64x2_t vzero = vdupq_n_s64(0);
                const uint64x2_t maskPos = vcgtq_s64(vdiff, vzero);
                const uint64x2_t maskNeg = vcgtq_s64(vzero, vdiff);

                const int64x2_t vpos = vbslq_s64(maskPos, vdiff, vzero);
                const int64x2_t vneg = vbslq_s64(maskNeg, vnegq_s64(vdiff), vzero);

                g_plus_unary_bits += static_cast<size_t>(vgetq_lane_s64(vpos, 0))
                        + static_cast<size_t>(vgetq_lane_s64(vpos, 1));
                g_minus_unary_bits += static_cast<size_t>(vgetq_lane_s64(vneg, 0))
                        + static_cast<size_t>(vgetq_lane_s64(vneg, 1));

                lastHighBits = static_cast<uint64_t>(hb1);
            }
#endif

            // Scalar tail (and full path when no SIMD)
            for (; i < N; ++i) {
                const T element = static_cast<T>(S[i] - base);
                const uint64_t currentHighBits = static_cast<uint64_t>(highPart(element, total_bits, h));
                const int64_t gap = static_cast<int64_t>(currentHighBits) - static_cast<int64_t>(lastHighBits);
                if (gap > 0) {
                    g_plus_unary_bits += static_cast<size_t>(gap);
                } else if (gap < 0) {
                    g_minus_unary_bits += static_cast<size_t>(-gap);
                }
                lastHighBits = currentHighBits;
            }

            // 2) Calculate the total size in bits for all data structures (unchanged)
            const size_t L_bits = N * b; // L stores low bits for all N elements
            const size_t G_plus_bits = g_plus_unary_bits + N; // N terminators
            const size_t G_minus_bits = g_minus_unary_bits + N; // N terminators

            size_t total_data_bits = L_bits + G_plus_bits + G_minus_bits;

            const double rank_overhead = factory->get_rank_overhead();
            const double select0_overhead = factory->get_select0_overhead();
            const size_t overhead_bits = static_cast<size_t>(
                std::ceil((G_plus_bits + G_minus_bits) * (rank_overhead + select0_overhead))
            );
            total_data_bits += overhead_bits;

            return total_data_bits;
        }

        static size_t evaluate_space(const std::vector<T> &S,
                                     uint8_t total_bits,
                                     uint8_t b,
                                     const std::shared_ptr<IBitVectorFactory> &factory) {
            const T base = *std::min_element(S.begin(), S.end());
            return evaluate_space(S, base, total_bits, b, factory);
        }

        static uint8_t binary_search_optimal_split_point(const std::vector<T> &S, const uint8_t total_bits,
                                                         const T min,
                                                         const T /*max*/,
                                                         const std::shared_ptr<IBitVectorFactory> &factory) {
            if (total_bits == 0) return 0;
            if (total_bits == 1) {
                size_t c0 = evaluate_space(S, min, total_bits, 0, factory);
                size_t c1 = evaluate_space(S, min, total_bits, 1, factory);
                return (c0 <= c1) ? 0 : 1;
            }

            const uint8_t lo0 = 0, hi0 = total_bits; // inclusive domain
            std::vector<size_t> cache(hi0 + 1, std::numeric_limits<size_t>::max());

            auto eval = [&](uint8_t b) -> size_t {
                size_t &ref = cache[b];
                if (ref == std::numeric_limits<size_t>::max()) {
                    ref = evaluate_space(S, min, total_bits, b, factory);
                }
                return ref;
            };

            // Optional: use your approximate heuristic to shrink the bracket
            // This is typically very effective and saves more evals:
            // uint8_t guess = approximate_optimal_split_point(S, total_bits, base, max_val);
            // uint8_t lo = (guess > 8) ? (guess - 8) : 0;
            // uint8_t hi = std::min<uint8_t>(total_bits, guess + 8);

            uint8_t lo = lo0, hi = hi0;
            while (hi - lo > 3) {
                uint8_t m1 = lo + (hi - lo) / 3;
                uint8_t m2 = hi - (hi - lo) / 3;
                size_t f1 = eval(m1);
                size_t f2 = eval(m2);
                if (f1 > f2) {
                    lo = static_cast<uint8_t>(m1 + 1);
                } else {
                    hi = static_cast<uint8_t>(m2 - 1);
                }
            }

            // Final scan of the tiny interval [lo..hi]
            uint8_t best_b = lo;
            size_t best_v = eval(lo);
            for (uint8_t b = static_cast<uint8_t>(lo + 1); b <= hi; ++b) {
                size_t v = eval(b);
                if (v < best_v) {
                    best_v = v;
                    best_b = b;
                }
            }
            return best_b;
        }

        static uint8_t approximate_optimal_split_point(const std::vector<T> &S,
                                                       uint8_t total_bits,
                                                       T /*min*/, T /*max*/) {
            const size_t n = S.size();
            if (n <= 1) return 0;

            const T *p = S.data();
            double sum = 0.0;
            size_t i = 1;

#if defined(__AVX2__)
            // AVX2 implementation: processes 4 doubles at a time
            __m256d sum_vec = _mm256_setzero_pd();
            const __m256d abs_mask = _mm256_set1_pd(-0.0); // Mask to clear the sign bit for abs

            for (; i + 3 < n; i += 4) {
                // Load 4 sets of previous/current values into vectors
                __m256d prev_vals = _mm256_set_pd((double)p[i + 2], (double)p[i + 1], (double)p[i], (double)p[i - 1]);
                __m256d curr_vals = _mm256_set_pd((double)p[i + 3], (double)p[i + 2], (double)p[i + 1], (double)p[i]);

                // Compute 4 differences in one instruction
                __m256d diffs = _mm256_sub_pd(curr_vals, prev_vals);

                // Compute absolute value for 4 differences
                __m256d abs_diffs = _mm256_andnot_pd(abs_mask, diffs);

                // Add to accumulator vector
                sum_vec = _mm256_add_pd(sum_vec, abs_diffs);
            }

            // Horizontal sum: add the 4 partial sums from the vector into the scalar `sum`
            double temp_sum[4];
            _mm256_storeu_pd(temp_sum, sum_vec);
            sum = temp_sum[0] + temp_sum[1] + temp_sum[2] + temp_sum[3];

#elif defined(__ARM_NEON) && defined(__aarch64__)
            // ARM NEON (AArch64) implementation: processes 2 doubles at a time
            float64x2_t sum_vec = vdupq_n_f64(0.0);

            for (; i + 1 < n; i += 2) {
                // NEON vector initialization is more direct
                float64x2_t prev_vals = {(double) p[i - 1], (double) p[i]};
                float64x2_t curr_vals = {(double) p[i], (double) p[i + 1]};

                // Compute 2 differences in one instruction
                float64x2_t diffs = vsubq_f64(curr_vals, prev_vals);

                // Compute absolute value for 2 differences
                float64x2_t abs_diffs = vabsq_f64(diffs);

                // Add to accumulator vector
                sum_vec = vaddq_f64(sum_vec, abs_diffs);
            }

            // Horizontal sum for NEON
            sum = vaddvq_f64(sum_vec);

#else
            // Fallback for other architectures: original unrolled loop
            for (; i + 3 < n; i += 4) {
                double d1 = static_cast<double>(p[i]) - static_cast<double>(p[i - 1]);
                double d2 = static_cast<double>(p[i + 1]) - static_cast<double>(p[i]);
                double d3 = static_cast<double>(p[i + 2]) - static_cast<double>(p[i + 1]);
                double d4 = static_cast<double>(p[i + 3]) - static_cast<double>(p[i + 2]);
                sum += std::abs(d1) + std::abs(d2) + std::abs(d3) + std::abs(d4);
            }
#endif

            // Remainder loop for all implementations
            for (; i < n; ++i) {
                double d = static_cast<double>(p[i]) - static_cast<double>(p[i - 1]);
                sum += std::abs(d);
            }

            const double avg_gap = sum / static_cast<double>(n);
            if (!(avg_gap > 1.0)) return 0;

            int exp;
            double m = std::frexp(avg_gap, &exp);
            uint8_t k = static_cast<uint8_t>(exp - (m == 0.5 ? 1 : 0));

            if (k > total_bits) k = total_bits;
            return k;
        }


        static uint8_t brute_force_optima_split_point(const std::vector<T> &S, const uint8_t total_bits, const T min,
                                                      const T /*max*/, const std::shared_ptr<IBitVectorFactory> &factory) {
            uint8_t best_split_point = 0;
            size_t best_space = std::numeric_limits<size_t>::max();
            for (uint8_t b = 0; b <= total_bits; b++) {
                const size_t space = evaluate_space(S, min, total_bits, b, factory);
                if (space < best_space) {
                    best_split_point = b;
                    best_space = space;
                }
            }
            return best_split_point;
        }

        static T highPart(const T x, const uint8_t total_bits, const uint8_t highBits) {
            const uint8_t lowBits = total_bits - highBits;
            return static_cast<T>(static_cast<std::make_unsigned_t<T>>(x) >> lowBits);
        }

        static T lowPart(const T x, const uint8_t lowBits) {
            if (lowBits >= sizeof(T) * 8) {
                return x;
            }
            const std::make_unsigned_t<T> mask = (static_cast<std::make_unsigned_t<T>>(1) << lowBits) - 1;
            return static_cast<T>(static_cast<std::make_unsigned_t<T>>(x) & mask);
        }

    public:
        using IGEF<T>::serialize;
        using IGEF<T>::load;

        ~B_GEF_NO_RLE() override = default;

        // Default constructor
        B_GEF_NO_RLE() : h(0), b(0), base(0) {
        }

        // 2. Copy Constructor
        B_GEF_NO_RLE(const B_GEF_NO_RLE &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              L(other.L),
              h(other.h),
              b(other.b),
              base(other.base) {
            if (other.h > 0) {
                G_plus = other.G_plus->clone();
                G_plus->enable_rank();
                G_plus->enable_select0();

                G_minus = other.G_minus->clone();
                G_minus->enable_rank();
                G_minus->enable_select0();
            } else {
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        // Friend swap function for copy-and-swap idiom
        friend void swap(B_GEF_NO_RLE &first, B_GEF_NO_RLE &second) noexcept {
            using std::swap;
            swap(first.L, second.L);
            swap(first.h, second.h);
            swap(first.b, second.b);
            swap(first.base, second.base);
            swap(first.G_plus, second.G_plus);
            swap(first.G_minus, second.G_minus);
        }

        // 3. Copy Assignment Operator (using copy-and-swap idiom)
        B_GEF_NO_RLE &operator=(const B_GEF_NO_RLE &other) {
            if (this != &other) {
                B_GEF_NO_RLE temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // 4. Move Constructor
        B_GEF_NO_RLE(B_GEF_NO_RLE &&other) noexcept
            : IGEF<T>(std::move(other)),
              G_plus(std::move(other.G_plus)),
              G_minus(std::move(other.G_minus)),
              L(std::move(other.L)),
              h(other.h),
              b(other.b),
              base(other.base) {
            // Leave the moved-from object in a valid, empty state
            other.h = 0;
            other.base = T{};
        }


        // 5. Move Assignment Operator
        B_GEF_NO_RLE &operator=(B_GEF_NO_RLE &&other) noexcept {
            if (this != &other) {
                G_plus = std::move(other.G_plus);
                G_minus = std::move(other.G_minus);
                L = std::move(other.L);
                h = other.h;
                b = other.b;
                base = other.base;
            }
            return *this;
        }


        // Constructor
        B_GEF_NO_RLE(const std::shared_ptr<IBitVectorFactory> &bit_vector_factory,
                     const std::vector<T> &S,
                     SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT) {
            const size_t N = S.size();
            if (N == 0) {
                b = 0;
                h = 0;
                base = T{};
                return;
            }

            auto [min_it, max_it] = std::minmax_element(S.begin(), S.end());
            base = *min_it;
            const T max_val = *max_it;
            const uint64_t u = max_val - base + 1;
            const uint8_t total_bits = (u > 1) ? static_cast<uint8_t>(floor(log2(u)) + 1) : 1;

            switch (strategy) {
                case BINARY_SEARCH_SPLIT_POINT:
                    b = binary_search_optimal_split_point(S, total_bits, base, max_val, bit_vector_factory);
                    break;
                case APPROXIMATE_SPLIT_POINT:
                    b = approximate_optimal_split_point(S, total_bits, base, max_val);
                    break;
                case BRUTE_FORCE_SPLIT_POINT:
                    b = brute_force_optima_split_point(S, total_bits, base, max_val, bit_vector_factory);
                    break;
            }
            h = total_bits - b;

            L = sdsl::int_vector<>(N, 0, b);
            if (h == 0) {
                // Special case: no high bits, only L is needed.
                for (size_t i = 0; i < N; ++i) {
                    L[i] = S[i] - base;
                }
                G_plus = nullptr;
                G_minus = nullptr;
                return;
            }

            // --- PASS 1: Analyze the sequence and determine exact sizes ---
            size_t g_plus_unary_bits = 0;
            size_t g_minus_unary_bits = 0;

            using U = std::make_unsigned_t<T>;
            T lastHighBits = 0;
            for (size_t i = 0; i < N; ++i) {
                const T element = S[i] - base;
                const T currentHighBits = static_cast<T>(static_cast<U>(element) >> b);

                const int64_t gap = static_cast<int64_t>(currentHighBits) - static_cast<int64_t>(lastHighBits);
                if (gap > 0) {
                    g_plus_unary_bits += static_cast<size_t>(gap);
                } else if (gap < 0) {
                    g_minus_unary_bits += static_cast<size_t>(-gap);
                }
                lastHighBits = currentHighBits;
            }

            const size_t g_plus_bits = g_plus_unary_bits + N;
            const size_t g_minus_bits = g_minus_unary_bits + N;

            // --- PASS 2: Allocate memory and populate structures ---
            G_plus = bit_vector_factory->create(g_plus_bits);
            G_minus = bit_vector_factory->create(g_minus_bits);

            size_t g_plus_pos = 0;
            size_t g_minus_pos = 0;
            lastHighBits = 0;

            const U low_mask = b ? (U(~U(0)) >> (sizeof(T) * 8 - b)) : U(0);

            if (b == 0) {
                for (size_t i = 0; i < N; ++i) {
                    const T element = S[i] - base;
                    const T currentHighBits = static_cast<T>(static_cast<U>(element) >> 0);

                    const int64_t gap = static_cast<int64_t>(currentHighBits) - static_cast<int64_t>(lastHighBits);

                    if (gap > 0) {
                        G_plus->set_range(g_plus_pos, static_cast<size_t>(gap), true);
                        g_plus_pos += static_cast<size_t>(gap);
                    } else {
                        const size_t neg = static_cast<size_t>(-gap);
                        G_minus->set_range(g_minus_pos, neg, true);
                        g_minus_pos += neg;
                    }
                    // Adding terminators
                    G_minus->set(g_minus_pos++, false);
                    G_plus->set(g_plus_pos++, false);

                    lastHighBits = currentHighBits;
                }
            } else {
                for (size_t i = 0; i < N; ++i) {
                    const T element = S[i] - base;
                    L[i] = static_cast<T>(static_cast<U>(element) & low_mask);

                    const T currentHighBits = static_cast<T>(static_cast<U>(element) >> b);
                    const int64_t gap = static_cast<int64_t>(currentHighBits) - static_cast<int64_t>(lastHighBits);

                    if (gap > 0) {
                        G_plus->set_range(g_plus_pos, static_cast<size_t>(gap), true);
                        g_plus_pos += static_cast<size_t>(gap);
                    } else {
                        // gap <= 0
                        const size_t neg = static_cast<size_t>(-gap);
                        G_minus->set_range(g_minus_pos, neg, true);
                        g_minus_pos += neg;
                    }
                    // Adding terminators
                    G_minus->set(g_minus_pos++, false);
                    G_plus->set(g_plus_pos++, false);

                    lastHighBits = currentHighBits;
                }
            }

            // Enable rank/select support
            G_plus->enable_rank();
            G_plus->enable_select0();
            G_minus->enable_rank();
            G_minus->enable_select0();
        }

        T operator[](size_t index) const override {
            if (h == 0)
                return base + L[index];

            const T base_high_val = 0;
            const size_t pos_gap = G_plus->rank(G_plus->select0(index + 1));
            const size_t neg_gap = G_minus->rank(G_minus->select0(index + 1));
            const T high_val = base_high_val + pos_gap - neg_gap;

            return base + (L[index] | (high_val << b));
        }

        void serialize(std::ofstream &ofs) const override {
            if (!ofs.is_open()) {
                throw std::runtime_error("Could not open file for serialization");
            }
            ofs.write(reinterpret_cast<const char *>(&h), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&b), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&base), sizeof(T));
            L.serialize(ofs);
            if (h > 0) {
                G_plus->serialize(ofs);
                G_minus->serialize(ofs);
            }
        }

        void load(std::ifstream &ifs, const std::shared_ptr<IBitVectorFactory> bit_vector_factory) override {
            ifs.read(reinterpret_cast<char *>(&h), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&b), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&base), sizeof(T));
            L.load(ifs);
            if (h > 0) {
                G_plus = bit_vector_factory->from_stream(ifs);
                G_plus->enable_rank();
                G_plus->enable_select0();

                G_minus = bit_vector_factory->from_stream(ifs);
                G_minus->enable_rank();
                G_minus->enable_select0();
            } else {
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return L.size();
        }

        [[nodiscard]] size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (h > 0) {
                total_bytes += G_plus->size_in_bytes();
                total_bytes += G_minus->size_in_bytes();
            }
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            return total_bytes;
        }

        [[nodiscard]] uint8_t split_point() const override {
            return this->b;
        }
    };
} // namespace gef

#endif