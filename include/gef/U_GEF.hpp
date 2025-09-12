//
// Created by Michelangelo Pucci on 06/07/25.
//

#ifndef U_GEF_HPP
#define U_GEF_HPP

#include <iostream>
#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits> // Required for std::make_unsigned
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
    class U_GEF : public IGEF<T> {
    private:
        // Bit-vector such that B[i] = 0 <==> 0 <= highPart(i) - highPart(i - 1) <= h
        std::unique_ptr<IBitVector> B;

        /*
        * Bit-vector that store the gaps between consecutive high-parts
        * such that 0 <= highPart(i) - highPart(i - 1) <= h
        */
        std::unique_ptr<IBitVector> G;

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

        static size_t evaluate_space(const std::vector<T> &S, const uint8_t total_bits, uint8_t b,
                                     const std::shared_ptr<IBitVectorFactory> &factory) {
            const size_t N = S.size();
            if (N == 0) {
                return sizeof(T) + sizeof(uint8_t) * 2; // Overhead for base, h, b
            }

            // Handle edge cases for the split point 'b'
            if (b == 0) {
                return std::numeric_limits<size_t>::max();
            }
            if (b >= total_bits) {
                // In this case, only the L vector exists, storing the full values (h=0).
                return N * total_bits;
            }

            const T base = *std::min_element(S.begin(), S.end());
            const uint8_t h = total_bits - b;

            size_t num_exceptions = 0;
            size_t g_unary_bits = 0;
            uint64_t lastHighBits = 0;

            size_t i = 0;

            if (N > 0) {
                const uint64_t hb0 = static_cast<uint64_t>(highPart(static_cast<T>(S[0] - base), total_bits, h));
                num_exceptions++;
                lastHighBits = hb0;
                i = 1;
            }

#if defined(__AVX2__)
// Process 4 elements per iteration
for (; i + 4 <= N; i += 4) {
const uint64_t hb0 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 0] - base), total_bits, h));
const uint64_t hb1 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 1] - base), total_bits, h));
const uint64_t hb2 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 2] - base), total_bits, h));
const uint64_t hb3 = static_cast<uint64_t>(highPart(static_cast<T>(S[i + 3] - base), total_bits, h));

alignas(32) uint64_t tmpCurr[4] = { hb0, hb1, hb2, hb3 };
const __m256i vcurr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpCurr));

const __m128i prev_lo = _mm_set_epi64x(static_cast<long long>(hb0), static_cast<long long>(lastHighBits));
const __m128i prev_hi = _mm_set_epi64x(static_cast<long long>(hb2), static_cast<long long>(hb1));
const __m256i vprev = _mm256_set_m128i(prev_hi, prev_lo);

const __m256i vdiff = _mm256_sub_epi64(vcurr, vprev);

const __m256i vzero = _mm256_setzero_si256();
const __m256i maskPos = _mm256_cmpgt_epi64(vdiff, vzero);
const __m256i maskNeg = _mm256_cmpgt_epi64(vzero, vdiff);

const __m128i vh_lo = _mm_set_epi64x(static_cast<long long>(h - 1), static_cast<long long>(h - 1));
const __m128i vh_hi = _mm_set_epi64x(static_cast<long long>(h - 1), static_cast<long long>(h - 1));
const __m256i vHminus1 = _mm256_set_m128i(vh_hi, vh_lo);
const __m256i maskGeH = _mm256_cmpgt_epi64(vdiff, vHminus1);

const __m256i vpos = _mm256_blendv_epi8(vzero, vdiff, maskPos);
const __m256i vposWithin = _mm256_blendv_epi8(vpos, vzero, maskGeH);

alignas(32) int64_t pos[4];
_mm256_storeu_si256(reinterpret_cast<__m256i*>(pos), vposWithin);

g_unary_bits += static_cast<size_t>(pos[0]) + static_cast<size_t>(pos[1]) +
static_cast<size_t>(pos[2]) + static_cast<size_t>(pos[3]);

const __m128i vone_lo = _mm_set_epi64x(1, 1);
const __m128i vone_hi = _mm_set_epi64x(1, 1);
const __m256i vone = _mm256_set_m128i(vone_hi, vone_lo);

const __m256i vexcNeg = _mm256_blendv_epi8(vzero, vone, maskNeg);
const __m256i vexcGeH = _mm256_blendv_epi8(vzero, vone, maskGeH);

alignas(32) int64_t excNeg[4], excGeH[4];
_mm256_storeu_si256(reinterpret_cast<__m256i*>(excNeg), vexcNeg);
_mm256_storeu_si256(reinterpret_cast<__m256i*>(excGeH), vexcGeH);

num_exceptions += static_cast<size_t>(excNeg[0] + excNeg[1] + excNeg[2] + excNeg[3] +
excGeH[0] + excGeH[1] + excGeH[2] + excGeH[3]);

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

const __m128i vHminus1 = _mm_set_epi64x(static_cast<long long>(h - 1), static_cast<long long>(h - 1));
const __m128i maskGeH = _mm_cmpgt_epi64(vdiff, vHminus1);

const __m128i vpos = _mm_blendv_epi8(vzero, vdiff, maskPos);
const __m128i vposWithin = _mm_blendv_epi8(vpos, vzero, maskGeH);

alignas(16) int64_t pos[2];
_mm_storeu_si128(reinterpret_cast<__m128i*>(pos), vposWithin);

g_unary_bits += static_cast<size_t>(pos[0]) + static_cast<size_t>(pos[1]);

const __m128i vone = _mm_set_epi64x(1, 1);
const __m128i vexcNeg = _mm_blendv_epi8(vzero, vone, maskNeg);
const __m128i vexcGeH = _mm_blendv_epi8(vzero, vone, maskGeH);

alignas(16) int64_t excNeg[2], excGeH[2];
_mm_storeu_si128(reinterpret_cast<__m128i*>(excNeg), vexcNeg);
_mm_storeu_si128(reinterpret_cast<__m128i*>(excGeH), vexcGeH);

num_exceptions += static_cast<size_t>(excNeg[0] + excNeg[1] + excGeH[0] + excGeH[1]);

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

                const int64x2_t vHminus1 = vdupq_n_s64(static_cast<int64_t>(h - 1));
                const uint64x2_t maskGeH = vcgtq_s64(vdiff, vHminus1);

                const int64x2_t vposWithin = vbslq_s64(maskGeH, vzero, vpos);

                g_unary_bits += static_cast<size_t>(vgetq_lane_s64(vposWithin, 0)) +
                        static_cast<size_t>(vgetq_lane_s64(vposWithin, 1));

                const int64x2_t vone = vdupq_n_s64(1);
                const int64x2_t vexcNeg = vbslq_s64(maskNeg, vone, vzero);
                const int64x2_t vexcGeH = vbslq_s64(maskGeH, vone, vzero);

                num_exceptions += static_cast<size_t>(vgetq_lane_s64(vexcNeg, 0) + vgetq_lane_s64(vexcNeg, 1) +
                                                      vgetq_lane_s64(vexcGeH, 0) + vgetq_lane_s64(vexcGeH, 1));

                lastHighBits = static_cast<uint64_t>(hb1);
            }
#endif

            for (; i < N; ++i) {
                const T element = static_cast<T>(S[i] - base);
                const uint64_t currentHighBits = static_cast<uint64_t>(highPart(element, total_bits, h));
                const int64_t diff = static_cast<int64_t>(currentHighBits) - static_cast<int64_t>(lastHighBits);
                if (diff < 0 || diff >= h) {
                    num_exceptions++;
                } else {
                    g_unary_bits += static_cast<size_t>(diff);
                }
                lastHighBits = currentHighBits;
            }

            // 2. Calculate the total size in bits for all data structures
            const size_t L_bits = N * b; // L stores low bits for all N elements
            const size_t B_bits = N; // B has one bit per element
            const size_t H_bits = num_exceptions * h; // H stores h bits for each exception
            const size_t G_bits = g_unary_bits + N; // G stores unary gaps + N terminators

            size_t total_data_bits = L_bits + B_bits + H_bits + G_bits;
            const double rank_overhead = factory->get_rank_overhead();
            const double select1_overhead = factory->get_select1_overhead();
            const double select0_overhead = factory->get_select0_overhead();
            const size_t b_overhead = static_cast<size_t>(
                std::ceil(B_bits * (rank_overhead + select1_overhead))
            );
            const size_t g_overhead = static_cast<size_t>(
                std::ceil(G_bits * (rank_overhead + select0_overhead))
            );
            total_data_bits += b_overhead + g_overhead;
            return total_data_bits;
        }

        static uint8_t binary_search_optimal_split_point(const std::vector<T> &S, const uint8_t total_bits,
                                                         const T /*min*/,
                                                         const T /*max*/,
                                                         const std::shared_ptr<IBitVectorFactory> &factory) {
            if (total_bits <= 1) {
                // Handle trivial cases where a search is not possible.
                size_t space0 = evaluate_space(S, total_bits, 0, factory);
                if (total_bits == 0) return 0;
                size_t space1 = evaluate_space(S, total_bits, 1, factory);
                return (space0 < space1) ? 0 : 1;
            }

            uint8_t lo = 0, hi = total_bits;

            // Golden ratio constant, used to determine the probe points.
            // We use the reciprocal (1/phi) for interval reduction.
            const double inv_phi = (std::sqrt(5.0) - 1.0) / 2.0; // approx 0.618

            // Calculate the initial two interior points.
            uint8_t c = lo + static_cast<uint8_t>(std::round((hi - lo) * (1.0 - inv_phi)));
            uint8_t d = lo + static_cast<uint8_t>(std::round((hi - lo) * inv_phi));

            // Initial evaluations for the two points.
            size_t space_c = evaluate_space(S, total_bits, c, factory);
            size_t space_d = evaluate_space(S, total_bits, d, factory);

            while (c < d) {
                if (space_c < space_d) {
                    // The minimum is in the lower interval [lo, d].
                    // The old 'c' becomes the new 'd'.
                    hi = d - 1;
                    d = c;
                    space_d = space_c;

                    // We only need to compute a new 'c'.
                    c = lo + static_cast<uint8_t>(std::round((hi - lo) * (1.0 - inv_phi)));
                    space_c = evaluate_space(S, total_bits, c, factory);
                } else {
                    // The minimum is in the upper interval [c, hi].
                    // The old 'd' becomes the new 'c'.
                    lo = c + 1;
                    c = d;
                    space_c = space_d;

                    // We only need to compute a new 'd'.
                    d = lo + static_cast<uint8_t>(std::round((hi - lo) * inv_phi));
                    space_d = evaluate_space(S, total_bits, d, factory);
                }
            }

            // The minimum is at lo (or hi, which will be the same).
            return lo;
        }

        static uint8_t approximate_optimal_split_point(const std::vector<T> &S, const uint8_t total_bits, const T min,
                                                       const T max) {
            if (S.size() <= 1) {
                return 0;
            }

            size_t g = 0;
            size_t non_negatives = 0;

            size_t i = 1;

#if defined(__AVX2__)
// Process 4 differences per iteration
for (; i + 3 < S.size(); i += 4) {
const int64_t x0 = static_cast<int64_t>(S[i + 0]);
const int64_t x1 = static_cast<int64_t>(S[i + 1]);
const int64_t x2 = static_cast<int64_t>(S[i + 2]);
const int64_t x3 = static_cast<int64_t>(S[i + 3]);

alignas(32) int64_t currArr[4] = { x0, x1, x2, x3 };
const __m256i vcurr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(currArr));

const __m128i prev_lo = _mm_set_epi64x(static_cast<long long>(x0), static_cast<long long>(S[i - 1]));
const __m128i prev_hi = _mm_set_epi64x(static_cast<long long>(x2), static_cast<long long>(x1));
const __m256i vprev = _mm256_set_m128i(prev_hi, prev_lo);

const __m256i vdiff = _mm256_sub_epi64(vcurr, vprev);

const __m256i vzero = _mm256_setzero_si256();
const __m256i maskPos = _mm256_cmpgt_epi64(vdiff, vzero);
const __m256i maskNeg = _mm256_cmpgt_epi64(vzero, vdiff);

const __m256i vpos = _mm256_blendv_epi8(vzero, vdiff, maskPos);

alignas(32) int64_t pos[4];
alignas(32) int64_t neg[4];
_mm256_storeu_si256(reinterpret_cast<__m256i*>(pos), vpos);
_mm256_storeu_si256(reinterpret_cast<__m256i*>(neg), _mm256_blendv_epi8(vzero, _mm256_set1_epi64x(1), maskNeg));

g += static_cast<size_t>(pos[0]) + static_cast<size_t>(pos[1]) +
static_cast<size_t>(pos[2]) + static_cast<size_t>(pos[3]);

const size_t neg_cnt = static_cast<size_t>(neg[0] + neg[1] + neg[2] + neg[3]);
non_negatives += 4 - neg_cnt;
}
#elif defined(__SSE4_2__)
// Process 2 differences per iteration
for (; i + 1 < S.size(); i += 2) {
const int64_t x0 = static_cast<int64_t>(S[i + 0]);
const int64_t x1 = static_cast<int64_t>(S[i + 1]);

const __m128i vcurr = _mm_set_epi64x(static_cast<long long>(x1), static_cast<long long>(x0));
const __m128i vprev = _mm_set_epi64x(static_cast<long long>(x0), static_cast<long long>(S[i - 1]));

const __m128i vdiff = _mm_sub_epi64(vcurr, vprev);

const __m128i vzero = _mm_setzero_si128();
const __m128i maskPos = _mm_cmpgt_epi64(vdiff, vzero);
const __m128i maskNeg = _mm_cmpgt_epi64(vzero, vdiff);

const __m128i vpos = _mm_blendv_epi8(vzero, vdiff, maskPos);

alignas(16) int64_t pos[2];
alignas(16) int64_t neg[2];
_mm_storeu_si128(reinterpret_cast<__m128i*>(pos), vpos);
_mm_storeu_si128(reinterpret_cast<__m128i*>(neg), _mm_blendv_epi8(vzero, _mm_set_epi64x(1, 1), maskNeg));

g += static_cast<size_t>(pos[0]) + static_cast<size_t>(pos[1]);

const size_t neg_cnt = static_cast<size_t>(neg[0] + neg[1]);
non_negatives += 2 - neg_cnt;
}
#elif defined(__aarch64__) && defined(__ARM_NEON)
            // Process 2 differences per iteration
            for (; i + 1 < S.size(); i += 2) {
                const int64_t x0 = static_cast<int64_t>(S[i + 0]);
                const int64_t x1 = static_cast<int64_t>(S[i + 1]);

                int64x2_t vcurr = vsetq_lane_s64(x0, vdupq_n_s64(0), 0);
                vcurr = vsetq_lane_s64(x1, vcurr, 1);

                int64x2_t vprev = vsetq_lane_s64(static_cast<int64_t>(S[i - 1]), vdupq_n_s64(0), 0);
                vprev = vsetq_lane_s64(x0, vprev, 1);

                int64x2_t vdiff = vsubq_s64(vcurr, vprev);

                const int64x2_t vzero = vdupq_n_s64(0);
                const uint64x2_t maskPos = vcgtq_s64(vdiff, vzero);
                const uint64x2_t maskNeg = vcgtq_s64(vzero, vdiff);

                const int64x2_t vpos = vbslq_s64(maskPos, vdiff, vzero);

                g += static_cast<size_t>(vgetq_lane_s64(vpos, 0)) + static_cast<size_t>(vgetq_lane_s64(vpos, 1));

                const int64x2_t vone = vdupq_n_s64(1);
                const int64x2_t vneg = vbslq_s64(maskNeg, vone, vzero);
                const size_t neg_cnt = static_cast<size_t>(vgetq_lane_s64(vneg, 0) + vgetq_lane_s64(vneg, 1));
                non_negatives += 2 - neg_cnt;
            }
#endif

            for (; i < S.size(); i++) {
                if (S[i] >= S[i - 1]) {
                    g += static_cast<size_t>(S[i] - S[i - 1]);
                    non_negatives++;
                }
            }
            if (non_negatives == 0) {
                return 0;
            }
            double avg_gap = static_cast<double>(g) / non_negatives;
            if (avg_gap <= 0) {
                return 0;
            }
            return ceil(log2(avg_gap));
        }

        static uint8_t brute_force_optimal_split_point(const std::vector<T> &S, const uint8_t total_bits, const T min,
                                                       const T max,
                                                       const std::shared_ptr<IBitVectorFactory> &factory) {
            uint8_t best_split_point = 0;
            size_t best_space = evaluate_space(S, total_bits, best_split_point, factory);
            for (uint8_t b = 1; b <= total_bits; b++) {
                const size_t space = evaluate_space(S, total_bits, b, factory);
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

        ~U_GEF() override = default;

        // Default constructor
        U_GEF() : h(0), b(0), base(0) {
        }

        // 2. Copy Constructor
        U_GEF(const U_GEF &other)
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

                G = other.G->clone();
                G->enable_rank();
                G->enable_select0();
            } else {
                B = nullptr;
                G = nullptr;
            }
        }

        // Friend swap function for copy-and-swap idiom
        friend void swap(U_GEF &first, U_GEF &second) noexcept {
            using std::swap;
            swap(first.B, second.B);
            swap(first.H, second.H);
            swap(first.L, second.L);
            swap(first.h, second.h);
            swap(first.b, second.b);
            swap(first.base, second.base);
            swap(first.G, second.G);
        }

        // 3. Copy Assignment Operator (using copy-and-swap idiom)
        U_GEF &operator=(const U_GEF &other) {
            if (this != &other) {
                U_GEF temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // 4. Move Constructor
        U_GEF(U_GEF &&other) noexcept
            : IGEF<T>(std::move(other)),
              B(std::move(other.B)),
              G(std::move(other.G)),
              H(std::move(other.H)),
              L(std::move(other.L)),
              h(other.h),
              b(other.b),
              base(other.base) {
            // Leave the moved-from object in a valid, empty state
            other.h = 0;
            other.base = T{};
        }

        // 5. Move Assignment Operator
        U_GEF &operator=(U_GEF &&other) noexcept {
            if (this != &other) {
                B = std::move(other.B);
                G = std::move(other.G);
                H = std::move(other.H);
                L = std::move(other.L);
                h = other.h;
                b = other.b;
                base = other.base;
            }
            return *this;
        }

        // Constructor
        U_GEF(const std::shared_ptr<IBitVectorFactory> &bit_vector_factory,
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

            base = *std::min_element(S.begin(), S.end());
            const T max_val = *std::max_element(S.begin(), S.end());
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
                    b = brute_force_optimal_split_point(S, total_bits, base, max_val, bit_vector_factory);
                    break;
            }
            h = total_bits - b;

            L = sdsl::int_vector<>(N, 0, b);
            if (h == 0) {
                // Special case: no high bits, only L is needed.
                for (size_t i = 0; i < N; ++i) {
                    L[i] = S[i] - base;
                }
                B = nullptr;
                G = nullptr;
                H.resize(0);
                return;
            }

            // --- PASS 1: Analyze the sequence and determine exact sizes ---
            std::vector<bool> is_exception(N);
            std::vector<T> high_parts(N);
            size_t h_size = 0;
            size_t g_unary_bits = 0;

            T lastHighBits = 0;
            for (size_t i = 0; i < N; ++i) {
                const T element = S[i] - base;
                high_parts[i] = highPart(element, total_bits, h);

                const bool exception = (i == 0 || high_parts[i] < lastHighBits || high_parts[i] >= lastHighBits + h);
                is_exception[i] = exception;

                if (exception) {
                    h_size++;
                } else {
                    g_unary_bits += high_parts[i] - lastHighBits;
                }
                lastHighBits = high_parts[i];
            }

            const size_t g_bits = g_unary_bits + N - h_size;

            // --- PASS 2: Allocate memory and populate structures ---
            B = bit_vector_factory->create(N);
            H = sdsl::int_vector<>(h_size, 0, h);
            G = bit_vector_factory->create(g_bits);

            size_t h_idx = 0;
            size_t g_pos = 0;
            lastHighBits = 0;

            for (size_t i = 0; i < N; ++i) {
                const T element = S[i] - base;
                L[i] = lowPart(element, b);

                B->set(i, is_exception[i]);
                if (is_exception[i]) {
                    H[h_idx++] = high_parts[i];
                } else {
                    const T gap = high_parts[i] - lastHighBits;
                    G->set_range(g_pos, gap, true);
                    g_pos += gap;
                    // Adding terminators
                    G->set(g_pos++, false);
                }

                lastHighBits = high_parts[i];
            }

            // Enable rank/select support
            B->enable_rank();
            B->enable_select1();
            G->enable_rank();
            G->enable_select0();
        }

        T operator[](size_t index) const override {
            // Case 1: No high bits are used (h=0).
            // All information is stored in the L vector. Reconstruction is trivial.
            if (h == 0) {
                return base + L[index];
            }

            // Find the number of exceptions up to and including 'index'.
            // This determines the 'run' of non-exceptions 'index' belongs to and
            // gives us the correct index into the H vector for our base high value.
            const size_t run_index = B->rank(index + 1);
            const T base_high_val = H[run_index - 1];
            T high_val;

            // Case 2: The element at 'index' is an exception (B[index] == 1).
            // Its high part is stored explicitly in H. No further calculation is needed.
            if ((*B)[index]) {
                high_val = base_high_val;
            }
            // Case 3: The element is not an exception (B[index] == 0).
            // Its high part must be reconstructed by adding the sum of gaps within its
            // run to the base high value of the run's starting exception.
            else {
                // Find the start position of this run (i.e., the index of the last exception).
                const size_t run_start_pos = B->select(run_index);

                // To find the sum of gaps for this run, we use a cumulative sum approach.
                // Sum of gaps = (Cumulative gaps up to 'index') - (Cumulative gaps up to 'run_start_pos').

                // Calculate cumulative gaps up to 'index':
                // 1. Find the 1-based rank of the 0-bit in B at 'index'.
                const size_t zero_rank_at_index = (index + 1) - run_index;
                // 2. Find the total number of 1s in G before the corresponding terminator.
                const size_t total_gap_sum = G->rank(G->select0(zero_rank_at_index));

                // Calculate cumulative gaps up to the start of the run:
                // 1. Find the number of 0-bits in B that occurred before this run started.
                const size_t zeros_before_run = (run_start_pos + 1) - run_index;
                // 2. Find the sum of 1s in G up to that point.
                const size_t gap_sum_before_run = (zeros_before_run > 0) ? G->rank(G->select0(zeros_before_run)) : 0;

                // The sum of gaps specific to this run is the difference.
                const size_t gap_in_run = total_gap_sum - gap_sum_before_run;

                high_val = base_high_val + gap_in_run;
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
                G->serialize(ofs);
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
                G = bit_vector_factory->from_stream(ifs);
                G->enable_rank();
                G->enable_select0();
            } else {
                B = nullptr;
                G = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return L.size();
        }

        [[nodiscard]] size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (B) {
                total_bytes += B->size_in_bytes();
                total_bytes += G->size_in_bytes();
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
