//
// Created by Michelangelo Pucci on 25/07/25.
//

#ifndef B_GEF_HPP
#define B_GEF_HPP

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
    class B_GEF : public IGEF<T> {
    private:
        // Bit-vector such that B[i] = 0 <==> 0 <= highPart(i) - highPart(i - 1) <= h
        std::unique_ptr<IBitVector> B;

        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that 0 <= highPart(i) - highPart(i - 1) <= h
         */
        std::unique_ptr<IBitVector> G_plus;

        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that 0 <= highPart(i - 1) - highPart(i) <= h
         */
        std::unique_ptr<IBitVector> G_minus;

        // high parts
        sdsl::int_vector<> H;

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

        // Optimized evaluate_space: base-aware overload (used by search)
        static size_t evaluate_space(const std::vector<T> &S,
                                     const T &base,
                                     const uint8_t total_bits,
                                     const uint8_t b,
                                     const std::shared_ptr<IBitVectorFactory> &factory) {
            const size_t N = S.size();
            if (N == 0) {
                return sizeof(T) + sizeof(uint8_t) * 2; // Overhead for base, h, b
            }

            if (b >= total_bits) {
                // Only L is needed
                return N * total_bits;
            }

            if (b == 0) {
                // Disallow b == 0 for B_GEF objective (matches previous behavior)
                return std::numeric_limits<size_t>::max();
            }

            const uint8_t h = static_cast<uint8_t>(total_bits - b);

            size_t num_exceptions = 0;
            size_t g_plus_unary_bits = 0;
            size_t g_minus_unary_bits = 0;

            uint64_t lastHighBits = 0;
            size_t i = 0;

#if defined(__AVX2__)
            for (; i + 4 <= N; i += 4) {
                const uint64_t hb0 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 0] - base), total_bits, h));
                const uint64_t hb1 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 1] - base), total_bits, h));
                const uint64_t hb2 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 2] - base), total_bits, h));
                const uint64_t hb3 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 3] - base), total_bits, h));

                alignas(32) uint64_t tmpCurr[4] = { hb0, hb1, hb2, hb3 };
                const __m256i vcurr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpCurr));

                // vprev = [last, hb0, hb1, hb2]
                const __m128i prev_lo = _mm_set_epi64x(static_cast<long long>(hb0), static_cast<long long>(lastHighBits));
                const __m128i prev_hi = _mm_set_epi64x(static_cast<long long>(hb2), static_cast<long long>(hb1));
                const __m256i vprev = _mm256_set_m128i(prev_hi, prev_lo);

                const __m256i vdiff = _mm256_sub_epi64(vcurr, vprev);

                // Prepare masks
                const __m256i vzero = _mm256_setzero_si256();
                const __m256i maskPos = _mm256_cmpgt_epi64(vdiff, vzero);             // vdiff > 0
                const __m256i maskNeg = _mm256_cmpgt_epi64(vzero, vdiff);             // vdiff < 0

                // Absolute value: abs(vdiff) = (vdiff < 0) ? -vdiff : vdiff
                const __m256i vdiff_neg = _mm256_sub_epi64(vzero, vdiff);
                const __m256i vabs = _mm256_blendv_epi8(vdiff, vdiff_neg, maskNeg);

                const __m256i vh = _mm256_set1_epi64x(static_cast<long long>(h));
                __m256i maskExc = _mm256_cmpgt_epi64(vabs, vh);                        // abs(gap) > h

                // Force i==0 lane as exception in the very first block
                if (i == 0) {
                    alignas(32) uint64_t lane_mask[4] = { ~0ULL, 0ULL, 0ULL, 0ULL };
                    const __m256i first_lane = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_mask));
                    maskExc = _mm256_or_si256(maskExc, first_lane);
                }

                // Non-exception mask
                const __m256i allones = _mm256_cmpeq_epi64(vzero, vzero);
                const __m256i maskNonExc = _mm256_andnot_si256(maskExc, allones);

                // Positive and negative contributions (zero where not applicable)
                const __m256i vpos = _mm256_and_si256(_mm256_blendv_epi8(vzero, vdiff, maskPos), maskNonExc);
                const __m256i vneg = _mm256_and_si256(_mm256_blendv_epi8(vzero, vdiff_neg, maskNeg), maskNonExc);

                alignas(32) int64_t pos[4];
                alignas(32) int64_t neg[4];
                alignas(32) uint64_t exc[4];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(pos), vpos);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(neg), vneg);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(exc), maskExc);

                g_plus_unary_bits  += static_cast<size_t>(pos[0]) + static_cast<size_t>(pos[1])
                                    + static_cast<size_t>(pos[2]) + static_cast<size_t>(pos[3]);
                g_minus_unary_bits += static_cast<size_t>(neg[0]) + static_cast<size_t>(neg[1])
                                    + static_cast<size_t>(neg[2]) + static_cast<size_t>(neg[3]);

                num_exceptions += (exc[0] ? 1u : 0u) + (exc[1] ? 1u : 0u)
                                + (exc[2] ? 1u : 0u) + (exc[3] ? 1u : 0u);

                lastHighBits = hb3;
            }
#elif defined(__SSE4_2__)
            for (; i + 2 <= N; i += 2) {
                const uint64_t hb0 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 0] - base), total_bits, h));
                const uint64_t hb1 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 1] - base), total_bits, h));

                const __m128i vcurr = _mm_set_epi64x(static_cast<long long>(hb1), static_cast<long long>(hb0));
                const __m128i vprev = _mm_set_epi64x(static_cast<long long>(hb0), static_cast<long long>(lastHighBits));

                const __m128i vdiff = _mm_sub_epi64(vcurr, vprev);

                const __m128i vzero = _mm_setzero_si128();
                const __m128i maskPos = _mm_cmpgt_epi64(vdiff, vzero);
                const __m128i maskNeg = _mm_cmpgt_epi64(vzero, vdiff);

                const __m128i vdiff_neg = _mm_sub_epi64(vzero, vdiff);
                const __m128i vabs = _mm_blendv_epi8(vdiff, vdiff_neg, maskNeg);

                const __m128i vh = _mm_set1_epi64x(static_cast<long long>(h));
                __m128i maskExc = _mm_cmpgt_epi64(vabs, vh);

                if (i == 0) {
                    const __m128i first_lane = _mm_set_epi64x(0LL, -1LL);
                    maskExc = _mm_or_si128(maskExc, first_lane);
                }

                const __m128i allones = _mm_cmpeq_epi64(vzero, vzero);
                const __m128i maskNonExc = _mm_andnot_si128(maskExc, allones);

                const __m128i vpos = _mm_and_si128(_mm_blendv_epi8(vzero, vdiff, maskPos), maskNonExc);
                const __m128i vneg = _mm_and_si128(_mm_blendv_epi8(vzero, vdiff_neg, maskNeg), maskNonExc);

                alignas(16) int64_t pos[2];
                alignas(16) int64_t neg[2];
                alignas(16) uint64_t exc[2];
                _mm_storeu_si128(reinterpret_cast<__m128i*>(pos), vpos);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(neg), vneg);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(exc), maskExc);

                g_plus_unary_bits  += static_cast<size_t>(pos[0]) + static_cast<size_t>(pos[1]);
                g_minus_unary_bits += static_cast<size_t>(neg[0]) + static_cast<size_t>(neg[1]);

                num_exceptions += (exc[0] ? 1u : 0u) + (exc[1] ? 1u : 0u);

                lastHighBits = hb1;
            }
#elif defined(__aarch64__) && defined(__ARM_NEON)
            for (; i + 2 <= N; i += 2) {
                const int64_t hb0 = static_cast<int64_t>(highPart(static_cast<T>(S[i + 0] - base), total_bits, h));
                const int64_t hb1 = static_cast<int64_t>(highPart(static_cast<T>(S[i + 1] - base), total_bits, h));

                int64x2_t vcurr = vdupq_n_s64(0);
                vcurr = vsetq_lane_s64(hb0, vcurr, 0);
                vcurr = vsetq_lane_s64(hb1, vcurr, 1);

                int64x2_t vprev = vdupq_n_s64(0);
                vprev = vsetq_lane_s64(static_cast<int64_t>(lastHighBits), vprev, 0);
                vprev = vsetq_lane_s64(hb0, vprev, 1);

                const int64x2_t vdiff = vsubq_s64(vcurr, vprev);

                const int64x2_t vzero = vdupq_n_s64(0);
                const uint64x2_t maskPos = vcgtq_s64(vdiff, vzero);
                const uint64x2_t maskNeg = vcgtq_s64(vzero, vdiff);

                const int64x2_t vdiff_neg = vnegq_s64(vdiff);
                const int64x2_t vabs = vbslq_s64(maskNeg, vdiff_neg, vdiff);

                const int64x2_t vh = vdupq_n_s64(static_cast<int64_t>(h));
                uint64x2_t maskExc = vcgtq_s64(vabs, vh);

                if (i == 0) {
                    const uint64x2_t first_lane = { ~0ULL, 0ULL };
                    maskExc = vorrq_u64(maskExc, first_lane);
                }

                const uint64x2_t all_ones = vdupq_n_u64(~0ULL);
                const uint64x2_t maskNonExc = veorq_u64(maskExc, all_ones);

                const int64x2_t vpos = vbslq_s64(maskPos, vdiff, vzero);
                const int64x2_t vneg = vbslq_s64(maskNeg, vdiff_neg, vzero);

                const int64x2_t vpos_masked = vbslq_s64(maskNonExc, vpos, vzero);
                const int64x2_t vneg_masked = vbslq_s64(maskNonExc, vneg, vzero);

                g_plus_unary_bits  += static_cast<size_t>(vgetq_lane_s64(vpos_masked, 0))
                                    + static_cast<size_t>(vgetq_lane_s64(vpos_masked, 1));
                g_minus_unary_bits += static_cast<size_t>(vgetq_lane_s64(vneg_masked, 0))
                                    + static_cast<size_t>(vgetq_lane_s64(vneg_masked, 1));

                num_exceptions     += (vgetq_lane_u64(maskExc, 0) ? 1u : 0u)
                                    +  (vgetq_lane_u64(maskExc, 1) ? 1u : 0u);

                lastHighBits = static_cast<uint64_t>(hb1);
            }
#endif
            // Scalar tail
            for (; i < N; ++i) {
                const T element = static_cast<T>(S[i] - base);
                const uint64_t currentHighBits = static_cast<uint64_t>(highPart(element, total_bits, h));
                const int64_t gap = static_cast<int64_t>(currentHighBits) - static_cast<int64_t>(lastHighBits);

                const bool exception = (i == 0) || (std::llabs(gap) > h);
                if (exception) {
                    ++num_exceptions;
                } else {
                    if (gap > 0) {
                        g_plus_unary_bits += static_cast<size_t>(gap);
                    } else if (gap < 0) {
                        g_minus_unary_bits += static_cast<size_t>(-gap);
                    }
                }

                lastHighBits = currentHighBits;
            }

            // Total bit-size
            const size_t L_bits = N * b;
            const size_t B_bits = N;
            const size_t H_bits = static_cast<size_t>(num_exceptions) * h;
            const size_t G_plus_bits = g_plus_unary_bits + N;
            const size_t G_minus_bits = g_minus_unary_bits + N;

            size_t total_data_bits = L_bits + B_bits + H_bits + G_plus_bits + G_minus_bits;
            const double rank_overhead = factory->get_rank_overhead();
            const double select1_overhead = factory->get_select1_overhead();
            const double select0_overhead = factory->get_select0_overhead();
            const size_t b_overhead = static_cast<size_t>(
                std::ceil(B_bits * (rank_overhead + select1_overhead))
            );
            const size_t g_overhead = static_cast<size_t>(
                std::ceil((G_plus_bits + G_minus_bits) * (rank_overhead + select0_overhead))
            );
            total_data_bits += b_overhead + g_overhead;
            return total_data_bits;
        }

        // Wrapper: computes base once and forwards to the optimized core
        static size_t evaluate_space(const std::vector<T> &S,
                                     const uint8_t total_bits,
                                     uint8_t b,
                                     const std::shared_ptr<IBitVectorFactory> &factory) {
            if (S.empty()) {
                return sizeof(T) + sizeof(uint8_t) * 2;
            }
            const T base = *std::min_element(S.begin(), S.end());
            return evaluate_space(S, base, total_bits, b, factory);
        }

        static uint8_t binary_search_optimal_split_point(const std::vector<T> &S, const uint8_t total_bits,
                                                         const T min,
                                                         const T /*max*/,
                                                         const std::shared_ptr<IBitVectorFactory> &factory) {
            if (total_bits <= 1) {
                size_t space0 = evaluate_space(S, min, total_bits, 0, factory);
                if (total_bits == 0) return 0;
                size_t space1 = evaluate_space(S, min, total_bits, 1, factory);
                return (space0 < space1) ? 0 : 1;
            }

            // Memoize evaluations to avoid recomputing
            std::vector<size_t> cache(total_bits + 1, std::numeric_limits<size_t>::max());
            auto eval = [&](uint8_t bb) -> size_t {
                size_t &ref = cache[bb];
                if (ref == std::numeric_limits<size_t>::max()) {
                    ref = evaluate_space(S, min, total_bits, bb, factory);
                }
                return ref;
            };

            uint8_t lo = 0, hi = total_bits;

            const double inv_phi = (std::sqrt(5.0) - 1.0) / 2.0; // ~0.618

            uint8_t c = static_cast<uint8_t>(lo + std::round((hi - lo) * (1.0 - inv_phi)));
            uint8_t d = static_cast<uint8_t>(lo + std::round((hi - lo) * inv_phi));

            size_t space_c = eval(c);
            size_t space_d = eval(d);

            while (c < d) {
                if (space_c < space_d) {
                    hi = static_cast<uint8_t>(d - 1);
                    d = c;
                    space_d = space_c;

                    c = static_cast<uint8_t>(lo + std::round((hi - lo) * (1.0 - inv_phi)));
                    space_c = eval(c);
                } else {
                    lo = static_cast<uint8_t>(c + 1);
                    c = d;
                    space_c = space_d;

                    d = static_cast<uint8_t>(lo + std::round((hi - lo) * inv_phi));
                    space_d = eval(d);
                }
            }

            return lo;
        }

        static uint8_t approximate_optimal_split_point(const std::vector<T> &S, const uint8_t total_bits, const T min,
                                                       const T max) {
            if (S.size() <= 1) {
                return 0;
            }
            size_t g = 0;
            for (size_t i = 1; i < S.size(); i++) {
                g += S[i] >= S[i - 1] ? static_cast<size_t>(S[i] - S[i - 1])
                                      : static_cast<size_t>(S[i - 1] - S[i]);
            }
            double avg_gap = static_cast<double>(g) / S.size();
            if (avg_gap <= 0) {
                return 0;
            }
            return static_cast<uint8_t>(ceil(log2(avg_gap)));
        }

        static uint8_t brute_force_optima_split_point(const std::vector<T> &S, const uint8_t total_bits, const T min,
                                                      const T /*max*/,
                                                      const std::shared_ptr<IBitVectorFactory> &factory) {
            uint8_t best_split_point = 0;
            size_t best_space = evaluate_space(S, min, total_bits, best_split_point, factory);
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

        ~B_GEF() override = default;

        // Default constructor
        B_GEF() : h(0), b(0), base(0) {
        }

        // 2. Copy Constructor
        B_GEF(const B_GEF &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              H(other.H),
              L(other.L),
              h(other.h),
              b(other.b),
              base(other.base) {
            if (other.h > 0) {
                B = other.B->clone();
                B->enable_rank();
                B->enable_select1();

                G_plus = other.G_plus->clone();
                G_plus->enable_rank();
                G_plus->enable_select0();

                G_minus = other.G_minus->clone();
                G_minus->enable_rank();
                G_minus->enable_select0();
            } else {
                B = nullptr;
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        // Friend swap function for copy-and-swap idiom
        friend void swap(B_GEF &first, B_GEF &second) noexcept {
            using std::swap;
            swap(first.B, second.B);
            swap(first.H, second.H);
            swap(first.L, second.L);
            swap(first.h, second.h);
            swap(first.b, second.b);
            swap(first.base, second.base);
            swap(first.G_plus, second.G_plus);
            swap(first.G_minus, second.G_minus);
        }

        // 3. Copy Assignment Operator (using copy-and-swap idiom)
        B_GEF &operator=(const B_GEF &other) {
            if (this != &other) {
                B_GEF temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // 4. Move Constructor
        B_GEF(B_GEF &&other) noexcept
            : IGEF<T>(std::move(other)),
              B(std::move(other.B)),
              G_plus(std::move(other.G_plus)),
              G_minus(std::move(other.G_minus)),
              H(std::move(other.H)),
              L(std::move(other.L)),
              h(other.h),
              b(other.b),
              base(other.base) {
            other.h = 0;
            other.base = T{};
        }

        // 5. Move Assignment Operator
        B_GEF &operator=(B_GEF &&other) noexcept {
            if (this != &other) {
                B = std::move(other.B);
                G_plus = std::move(other.G_plus);
                G_minus = std::move(other.G_minus);
                H = std::move(other.H);
                L = std::move(other.L);
                h = other.h;
                b = other.b;
                base = other.base;
            }
            return *this;
        }

        // Constructor
        B_GEF(const std::shared_ptr<IBitVectorFactory> &bit_vector_factory,
              const std::vector<T> &S,
              SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT) {
            const size_t N = S.size();
            if (N == 0) {
                b = 0;
                h = 0;
                base = T{};
                B = nullptr;
                return;
            }

            auto [min_it, max_it] = std::minmax_element(S.begin(), S.end());
            base = *min_it;
            const T max_val = *max_it;
            const uint64_t u = static_cast<uint64_t>(max_val - base) + 1ULL;
            const uint8_t total_bits = (u > 1) ? static_cast<uint8_t>(std::floor(std::log2(u)) + 1) : 1;

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
            h = static_cast<uint8_t>(total_bits - b);

            L = sdsl::int_vector<>(N, 0, b);
            if (h == 0) {
                // Special case: no high bits, only L is needed.
                for (size_t i = 0; i < N; ++i) {
                    L[i] = S[i] - base;
                }
                B = nullptr;
                G_plus = nullptr;
                G_minus = nullptr;
                H.resize(0);
                return;
            }

            // --- PASS 1: Analyze the sequence and determine exact sizes ---
            std::vector<bool> is_exception(N);
            std::vector<T> high_parts(N);
            size_t h_size = 0;
            size_t g_plus_unary_bits = 0;
            size_t g_minus_unary_bits = 0;

            T lastHighBits = 0;
            for (size_t i = 0; i < N; ++i) {
                const T element = S[i] - base;
                high_parts[i] = highPart(element, total_bits, h);

                const int64_t gap = static_cast<int64_t>(high_parts[i]) - static_cast<int64_t>(lastHighBits);
                const bool exception = (i == 0 || std::llabs(gap) > h);
                is_exception[i] = exception;

                if (exception) {
                    h_size++;
                } else {
                    if (gap > 0) {
                        g_plus_unary_bits += static_cast<size_t>(gap);
                    } else { // gap <= 0
                        g_minus_unary_bits += static_cast<size_t>(-gap);
                    }
                }
                lastHighBits = high_parts[i];
            }

            const size_t g_plus_bits = g_plus_unary_bits + N - h_size;
            const size_t g_minus_bits = g_minus_unary_bits + N - h_size;

            // --- PASS 2: Allocate memory and populate structures ---
            B = bit_vector_factory->create(N);
            H = sdsl::int_vector<>(h_size, 0, h);
            G_plus = bit_vector_factory->create(g_plus_bits);
            G_minus = bit_vector_factory->create(g_minus_bits);

            size_t h_idx = 0;
            size_t g_plus_pos = 0;
            size_t g_minus_pos = 0;
            lastHighBits = 0;

            for (size_t i = 0; i < N; ++i) {
                const T element = S[i] - base;
                L[i] = lowPart(element, b);

                B->set(i, is_exception[i]);
                if (is_exception[i]) {
                    H[h_idx++] = high_parts[i];
                } else {
                    const int64_t gap = static_cast<int64_t>(high_parts[i]) - static_cast<int64_t>(lastHighBits);

                    if (gap > 0) {
                        G_plus->set_range(g_plus_pos, static_cast<size_t>(gap), true);
                        g_plus_pos += static_cast<size_t>(gap);
                    } else { // gap < 0
                        const size_t ng = static_cast<size_t>(-gap);
                        G_minus->set_range(g_minus_pos, ng, true);
                        g_minus_pos += ng;
                    }

                    // Terminators
                    G_minus->set(g_minus_pos++, false);
                    G_plus->set(g_plus_pos++, false);
                }

                lastHighBits = high_parts[i];
            }

            // Enable rank/select support
            B->enable_rank();
            B->enable_select1();
            G_plus->enable_rank();
            G_plus->enable_select0();
            G_minus->enable_rank();
            G_minus->enable_select0();
        }

        T operator[](size_t index) const override {
            // Case 1: No high bits are used (h=0).
            // The value is fully stored in the L vector.
            if (h == 0) {
                return base + L[index];
            }

            // Find the number of exceptions up to and including 'index'.
            // This identifies the 'run' of non-exceptions 'index' belongs to and
            // provides the index into H for the run's base high value.
            const size_t run_index = B->rank(index + 1);
            const T base_high_val = H[run_index - 1];
            T high_val;

            // Case 2: The element at 'index' is an exception (B[index] == 1).
            // Its high part is stored explicitly in H. The net gap contribution is zero.
            if ((*B)[index]) {
                high_val = base_high_val;
            }
            // Case 3: The element is not an exception (B[index] == 0).
            // Its high part is reconstructed by adding the net sum of gaps (positive - negative)
            // within its run to the base high value.
            else {
                // Find the start position of this run (i.e., the index of the last exception).
                const size_t run_start_pos = B->select(run_index);

                // To find the sum of gaps for this run, we use a cumulative sum approach.
                // Net Gap = (Cumulative gaps up to 'index') - (Cumulative gaps up to 'run_start_pos').
                // This must be done for both positive and negative gap vectors.

                // 1. Find the rank of the 0-bit in B at 'index' and before the run.
                const size_t zero_rank_at_index = (index + 1) - run_index;
                const size_t zeros_before_run = (run_start_pos + 1) - run_index;

                // 2. Calculate the sum of positive gaps within the run.
                const size_t total_pos_gap = G_plus->rank(G_plus->select0(zero_rank_at_index));
                const size_t pos_gap_before_run = (zeros_before_run > 0) ? G_plus->rank(G_plus->select0(zeros_before_run)) : 0;
                const size_t pos_gap_in_run = total_pos_gap - pos_gap_before_run;

                // 3. Calculate the sum of negative gaps within the run.
                const size_t total_neg_gap = G_minus->rank(G_minus->select0(zero_rank_at_index));
                const size_t neg_gap_before_run = (zeros_before_run > 0) ? G_minus->rank(G_minus->select0(zeros_before_run)) : 0;
                const size_t neg_gap_in_run = total_neg_gap - neg_gap_before_run;

                // 4. The final high part is the base value plus the net gap.
                high_val = base_high_val + static_cast<T>(pos_gap_in_run) - static_cast<T>(neg_gap_in_run);
            }

            // Finally, combine the reconstructed high part with the low part from L
            // and add the base offset to get the original value.
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
            H.serialize(ofs);
            if (h > 0) {
                B->serialize(ofs);
                G_plus->serialize(ofs);
                G_minus->serialize(ofs);
            }
        }

        void load(std::ifstream &ifs, const std::shared_ptr<IBitVectorFactory> bit_vector_factory) override {
            ifs.read(reinterpret_cast<char *>(&h), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&b), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&base), sizeof(T));
            L.load(ifs);
            H.load(ifs);
            if (h > 0) {
                B = bit_vector_factory->from_stream(ifs);
                B->enable_rank();
                B->enable_select1();
                G_plus = bit_vector_factory->from_stream(ifs);
                G_plus->enable_rank();
                G_plus->enable_select0();

                G_minus = bit_vector_factory->from_stream(ifs);
                G_minus->enable_rank();
                G_minus->enable_select0();
            } else {
                B = nullptr;
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return L.size();
        }

        [[nodiscard]] size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (B) {
                total_bytes += B->size_in_bytes();
                total_bytes += G_plus->size_in_bytes();
                total_bytes += G_minus->size_in_bytes();
            }
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sdsl::size_in_bytes(H);
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