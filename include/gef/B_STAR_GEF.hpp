//
// Created by Michelangelo Pucci on 03/08/25.
//

#ifndef B_STAR_GEF_HPP
#define B_STAR_GEF_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits>
#include <utility>
#include <chrono>
#include <stdexcept>

#include <sys/mman.h>


// SIMD intrinsics for optimized bit operations
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
#include <immintrin.h>
#elif defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
#include <arm_neon.h>
#endif

#if __has_include(<experimental/simd>) && !defined(GEF_DISABLE_SIMD)
#include <experimental/simd>
namespace stdx = std::experimental;
#define GEF_EXPERIMENTAL_SIMD_ENABLED
#endif

#include "IGEF.hpp"
#include "AllocatedSize.hpp"
#include "BlockParameters.hpp"
#include "RLE_GEF.hpp"
#include "gap_computation_utils.hpp"
#include "CompressionProfile.hpp"
#include "FastUnaryDecoder.hpp"
#include "NoRandomAccess.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/SDSLBitVector.hpp"
#include "../datastructures/PastaBitVector.hpp"
#include "FastBitWriter.hpp"


namespace gef {
    namespace internal {
    struct BStarZeroByte {
        uint8_t count = 0;
        std::array<uint8_t, 8> positions{};
    };

    // Shared across value types and bitvector backends. Entries enumerate set
    // bits in a complemented gap byte, i.e. its zero delimiters in input order.
    inline constexpr auto B_STAR_ZERO_BYTES = [] {
        std::array<BStarZeroByte, 256> table{};
        for (unsigned mask = 0; mask < table.size(); ++mask)
            for (unsigned bit = 0; bit < 8; ++bit)
                if (mask & (1u << bit))
                    table[mask].positions[table[mask].count++] = bit;
        return table;
    }();

    template<typename T, typename GapBitVectorType = PastaGapBitVector, bool RandomAccess = true>
    class B_STAR_GEF : public IGEF<T> {
    private:
        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that highPart(i) >= highPart(i - 1)
         */
        std::unique_ptr<GapBitVectorType> G_plus;

        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that highPart(i - 1) >= highPart(i)
         */
        std::unique_ptr<GapBitVectorType> G_minus;

        // low parts
        sdsl::int_vector<> L;

        // The split point that rules which bits are stored in H and in L
        uint8_t b;
        uint8_t h;
        size_t m_num_elements;

        /**
         * The minimum of the encoded sequence, so that we store the shifted sequence
         * that falls in the range [0, max S - base]
         * This tricks may boost compression and allows us to implicitly store negative numbers
         */
        T base;

        static size_t evaluate_space(const GapComputation &gap_computation,
                                     const uint8_t b,
                                     const size_t total_bits,
                                     const size_t N) {
            if (N == 0) {
                return sizeof(T) + sizeof(uint8_t) * 2;
            }

            const auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };
            size_t total_bytes = sizeof(T) + sizeof(uint8_t) * 2;
            total_bytes += bits_to_bytes(N * static_cast<size_t>(b));

            if (b < total_bits) {
                total_bytes += bits_to_bytes(gap_computation.sum_of_positive_gaps + N);
                total_bytes += bits_to_bytes(gap_computation.sum_of_negative_gaps + N);
            }

            return total_bytes;
        }

        // The representation stores offsets from base, whose exact width can be
        // computed without floating-point rounding or a range+1 overflow.
        static uint8_t value_width(T min, T max) {
            using U = std::make_unsigned_t<T>;
            return static_cast<uint8_t>(std::bit_width(static_cast<U>(
                static_cast<U>(max) - static_cast<U>(min))));
        }

        static GapComputation bitpack_statistics(size_t n) {
            GapComputation result{};
            result.positive_gaps = n;
            result.zero_gaps = n ? n - 1 : 0;
            return result;
        }

        template<typename C>
        static double approximated_optimal_split_point(const C &S, const T min, const T max) {
            using U = std::make_unsigned_t<T>;
            const U range = static_cast<U>(max) - static_cast<U>(min);
            if (range == 0 || S.size() == 0) return 0.0;

            // All magnitudes are at most range. Proving the sum fits once lets
            // the common path use an ordinary, vectorizable word-sized sum.
            double total_variation;
            if (range <= SIZE_MAX / S.size()) {
                size_t variation = static_cast<U>(static_cast<U>(S[0]) - static_cast<U>(min));
                for (size_t i = 1; i < S.size(); ++i) {
                    const U current = static_cast<U>(S[i]);
                    const U previous = static_cast<U>(S[i - 1]);
                    variation += S[i] >= S[i - 1]
                        ? static_cast<U>(current - previous)
                        : static_cast<U>(previous - current);
                }
                total_variation = static_cast<double>(variation);
            } else {
                // Full-width integer inputs need a wider accumulator, not a
                // saturated approximation (whose two partial sums can wrap).
                unsigned __int128 variation = static_cast<U>(static_cast<U>(S[0]) - static_cast<U>(min));
                for (size_t i = 1; i < S.size(); ++i) {
                    const U current = static_cast<U>(S[i]);
                    const U previous = static_cast<U>(S[i - 1]);
                    variation += S[i] >= S[i - 1]
                        ? static_cast<U>(current - previous)
                        : static_cast<U>(previous - current);
                }
                total_variation = static_cast<double>(variation);
            }
            return total_variation > 0
                ? std::log2(std::log(2.0) * total_variation / S.size()) : 0.0;
        }

        template<typename C>
        static std::pair<uint8_t, GapComputation> approximate_optimal_split_point(const C &S,
            const T min, const T max) {
            using U = std::make_unsigned_t<T>;
            const uint8_t total_bits = value_width(min, max);
            if (total_bits == 0) return {0, bitpack_statistics(S.size())};

            const double approx_b = std::clamp(approximated_optimal_split_point(S, min, max),
                                                0.0, static_cast<double>(total_bits));
            const uint8_t floor_b = static_cast<uint8_t>(std::floor(approx_b));
            const uint8_t ceil_b = static_cast<uint8_t>(std::ceil(approx_b));
            if (floor_b == total_bits) return {total_bits, bitpack_statistics(S.size())};

            GapComputation candidates[2]{};
            const U maximum_high = static_cast<U>(static_cast<U>(max) - static_cast<U>(min)) >> floor_b;
            if (maximum_high <= (SIZE_MAX - S.size()) / S.size()) {
                // The bound also proves that the total absolute variation fits.
                // Accumulate just that magnitude for each adjacent split. Since
                // the first previous high part is zero, positive-negative=last;
                // recover both stream lengths without per-value direction counts.
                size_t variation_floor = 0;
                size_t variation_ceil = 0;
                U previous = 0;
                for (const T value : S) {
                    const U current = static_cast<U>(static_cast<U>(value) - static_cast<U>(min)) >> floor_b;
                    variation_floor += current >= previous ? current - previous : previous - current;
                    if (ceil_b != floor_b) {
                        const U current_ceil = current >> 1;
                        const U previous_ceil = previous >> 1;
                        variation_ceil += current_ceil >= previous_ceil ? current_ceil - previous_ceil : previous_ceil - current_ceil;
                    }
                    previous = current;
                }
                candidates[0].sum_of_negative_gaps = (variation_floor - previous) / 2;
                candidates[0].sum_of_positive_gaps = candidates[0].sum_of_negative_gaps + previous;
                if (ceil_b != floor_b) {
                    candidates[1].sum_of_negative_gaps = (variation_ceil - (previous >> 1)) / 2;
                    candidates[1].sum_of_positive_gaps = candidates[1].sum_of_negative_gaps + (previous >> 1);
                }
            } else {
                candidates[0] = variation_of_shifted_vec(S, min, max, floor_b, ExceptionRule::None);
                if (ceil_b != floor_b)
                    candidates[1] = variation_of_shifted_vec(S, min, max, ceil_b, ExceptionRule::None);
            }

            uint8_t best_b = floor_b;
            size_t best_space = evaluate_space(candidates[0], floor_b, total_bits, S.size());
            GapComputation best_stats = candidates[0];
            if (ceil_b != floor_b) {
                const size_t ceil_space = evaluate_space(candidates[1], ceil_b, total_bits, S.size());
                if (ceil_space < best_space) {
                    best_b = ceil_b;
                    best_space = ceil_space;
                    best_stats = candidates[1];
                }
            }
            // With every bit in L, both high-part sums are known to be zero.
            // Evaluating the bitpacking fallback does not require another scan.
            const auto bitpack_stats = bitpack_statistics(S.size());
            if (evaluate_space(bitpack_stats, total_bits, total_bits, S.size()) < best_space)
                return {total_bits, bitpack_stats};
            return {best_b, best_stats};
        }


        template<typename C>
        static std::pair<uint8_t, GapComputation>
        optimal_split_point(const C &S, const T min, const T max) {
            const size_t total_bits = value_width(min, max);
            if (total_bits == 0) return {0, bitpack_statistics(S.size())};

            const double approx_b = std::clamp(approximated_optimal_split_point(S, min, max),
                                                0.0, static_cast<double>(total_bits));
            const size_t min_b = static_cast<size_t>(std::max(0.0, floor(approx_b) - 1));
            const size_t max_b = static_cast<size_t>(std::min(static_cast<double>(total_bits), ceil(approx_b) + 3));

            const size_t count = max_b - min_b + 1;
            assert(count <= 6);
            std::array<GapComputation, 6> gap_computation{};
            using U = std::make_unsigned_t<T>;
            // min_b <= total_bits-1 for a nonconstant block, even at width 64.
            assert(min_b < sizeof(T) * 8);
            const U maximum_high = static_cast<U>(static_cast<U>(max) - static_cast<U>(min)) >> min_b;
            if (maximum_high <= (SIZE_MAX - S.size()) / S.size()) {
                // Every gap is at most maximum_high. This proves all absolute
                // sums and their delimiter counts fit without saturating adds.
                std::array<size_t, 6> variation{};
                U previous = 0;
                for (const T value : S) {
                    const U residual = static_cast<U>(static_cast<U>(value) - static_cast<U>(min));
                    U current = residual >> min_b;
                    U prior = previous;
                    for (size_t j = 0; j < count; ++j) {
                        variation[j] += current >= prior ? current - prior : prior - current;
                        current >>= 1;
                        prior >>= 1;
                    }
                    previous = residual >> min_b;
                }
                for (size_t j = 0; j < count; ++j) {
                    // Starting from zero, positive-negative equals the final
                    // high part. Subtract before dividing to avoid V+last overflow.
                    gap_computation[j].sum_of_negative_gaps = (variation[j] - previous) / 2;
                    gap_computation[j].sum_of_positive_gaps = gap_computation[j].sum_of_negative_gaps + previous;
                    previous >>= 1;
                }
            } else {
                const auto generic = total_variation_of_shifted_vec_with_multiple_shifts(
                    S, min, max, static_cast<uint8_t>(min_b), static_cast<uint8_t>(max_b), ExceptionRule::None);
                std::copy(generic.begin(), generic.end(), gap_computation.begin());
            }

            size_t best_index = 0;
            size_t best_space = SIZE_MAX;
            for (size_t i = 0; i < count; i++) {
                const uint8_t current_b = static_cast<uint8_t>(min_b + i);
                const size_t space = evaluate_space(gap_computation[i], current_b, total_bits, S.size());
                if (space < best_space) {
                    best_space = space;
                    best_index = i;
                }
            }

            const auto total_bits_stats = bitpack_statistics(S.size());
            const size_t total_bits_space =
                evaluate_space(total_bits_stats, static_cast<uint8_t>(total_bits), total_bits, S.size());

            if (total_bits_space < best_space) {
                return {static_cast<uint8_t>(total_bits), total_bits_stats};
            }

            return {static_cast<uint8_t>(min_b + best_index), gap_computation[best_index]};
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


        // Enumerate zero delimiters directly. The difference between their
        // positions in G+ and G- is the high value, so sequential decoding needs
        // neither unary-gap materialization nor a prefix-sum dependency.
        class ZeroPositionReader {
            const uint64_t* data_;
            size_t word_;
            uint64_t zeros_;
        public:
            ZeroPositionReader(const uint64_t* data, size_t start)
                : data_(data), word_(start >> 6), zeros_(~data_[word_]) {
                const unsigned offset = start & 63;
                if constexpr (GapBitVectorType::reverse_bit_order)
                    zeros_ &= ~uint64_t(0) >> offset;
                else
                    zeros_ &= ~uint64_t(0) << offset;
            }

            // The caller requests no more delimiters than remain in the codec.
            // Thus padding bits are never returned, and no sentinel word is read.
            void read(size_t* out, size_t count) {
                size_t i = 0;
                while (i < count) {
                    while (zeros_ == 0) zeros_ = ~data_[++word_];
                    const size_t base_position = word_ * 64;
                    // Expand a full delimiter word without per-value bit scans.
                    if (zeros_ == UINT64_MAX && count - i >= 64) {
                        for (size_t offset = 0; offset < 64; ++offset)
                            out[i + offset] = base_position + offset;
                        i += 64;
                        zeros_ = 0;
                        continue;
                    }
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD) && SIZE_MAX == UINT64_MAX
                    if constexpr (!GapBitVectorType::reverse_bit_order) {
                        const auto emit_byte = [&](unsigned shift) __attribute__((always_inline)) {
                            const auto& entry = B_STAR_ZERO_BYTES[(zeros_ >> shift) & 255];
                            // Eight slots fit inside this private scratch request.
                            // Only entry.count become final; later iterations
                            // overwrite the remaining initialized padding slots
                            // before read() returns all count valid positions.
                            // Load exactly the eight initialized offset bytes;
                            // adjacent table entries need not be readable here.
                            uint64_t packed;
                            std::memcpy(&packed, entry.positions.data(), sizeof(packed));
                            const __m128i bytes = _mm_cvtsi64_si128(std::bit_cast<int64_t>(packed));
                            const __m256i base = _mm256_set1_epi64x(
                                std::bit_cast<int64_t>(uint64_t(base_position + shift)));
                            const __m256i first = _mm256_add_epi64(base, _mm256_cvtepu8_epi64(bytes));
                            const __m256i second = _mm256_add_epi64(base,
                                _mm256_cvtepu8_epi64(_mm_srli_si128(bytes, 4)));
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), first);
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 4), second);
                            i += entry.count;
                        };
                        if (count - i >= 64) {
                            // Before byte j, at most 8*j positions are final.
                            // All eight temporary stores therefore fit in the
                            // 64 available slots, including for empty bytes.
                            [&]<size_t... Bytes>(std::index_sequence<Bytes...>) __attribute__((always_inline)) {
                                (emit_byte(8 * Bytes), ...);
                            }(std::make_index_sequence<8>{});
                            zeros_ = 0;
                            continue;
                        }
                        if (count - i >= 8) {
                            const unsigned shift = std::countr_zero(zeros_) & ~7u;
                            emit_byte(shift);
                            // shift is at most 56; the final partial request
                            // stays scalar, so padding delimiters are not used.
                            zeros_ &= ~(uint64_t(255) << shift);
                            continue;
                        }
                    }
#endif
                    do {
                        if constexpr (GapBitVectorType::reverse_bit_order) {
                            const unsigned bit = std::countl_zero(zeros_);
                            out[i++] = base_position + bit;
                            zeros_ ^= uint64_t(1) << (63 - bit);
                        } else {
                            out[i++] = base_position + std::countr_zero(zeros_);
                            zeros_ &= zeros_ - 1;
                        }
                    } while (zeros_ && i < count);
                }
            }
        };

        template<uint8_t SPLIT_POINT>
        static uint64_t read_low(const uint64_t* data, size_t index) {
            if constexpr (SPLIT_POINT == 0) {
                return 0;
            } else if constexpr (SPLIT_POINT == 64) {
                return data[index];
            } else {
                constexpr uint64_t mask = (uint64_t(1) << SPLIT_POINT) - 1;
                const size_t bit_position = index * SPLIT_POINT;
                const size_t word = bit_position >> 6;
                const unsigned offset = bit_position & 63;
                uint64_t value = data[word] >> offset;
                if constexpr (64 % SPLIT_POINT != 0) {
                    if (offset + SPLIT_POINT > 64)
                        value |= data[word + 1] << (64 - offset);
                }
                return value & mask;
            }
        }

        // At an element index divisible by 64, the next 64 low parts occupy
        // exactly SPLIT_POINT words. Constant indices remove per-value offset
        // and word-crossing decisions without changing the packed representation.
        template<uint8_t SPLIT_POINT, size_t... I>
        static void reconstruct_aligned64(const uint64_t* low_data,
                                          const size_t* plus_positions,
                                          const size_t* minus_positions,
                                          std::make_unsigned_t<T> base_u,
                                          T* output,
                                          std::index_sequence<I...>) {
            static_assert(SPLIT_POINT > 0 && SPLIT_POINT < sizeof(T) * 8);
            static_assert(sizeof...(I) == 64);
            using U = std::make_unsigned_t<T>;
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
            if constexpr (sizeof(T) == 8 && sizeof(size_t) == 8) {
                // Pack four lows directly into registers, avoiding a low-part
                // scratch array. Both position and output pointers may be unaligned.
                const __m256i base_vector = _mm256_set1_epi64x(std::bit_cast<int64_t>(base_u));
                ([&] {
                    if constexpr (I % 4 == 0) {
                        __m256i lows;
                        if constexpr (SPLIT_POINT <= 16) {
                            // Four lows fit in one 64-bit window, even when the
                            // window spans two packed words. Both words remain
                            // inside this complete 64-element input block.
                            constexpr size_t word = (I * SPLIT_POINT) / 64;
                            constexpr unsigned offset = (I * SPLIT_POINT) % 64;
                            uint64_t window = low_data[word] >> offset;
                            if constexpr (offset + 4 * SPLIT_POINT > 64)
                                window |= low_data[word + 1] << (64 - offset);
                            const __m256i shifts = _mm256_set_epi64x(3 * SPLIT_POINT, 2 * SPLIT_POINT, SPLIT_POINT, 0);
                            const __m256i mask = _mm256_set1_epi64x((uint64_t(1) << SPLIT_POINT) - 1);
                            lows = _mm256_and_si256(_mm256_srlv_epi64(
                                _mm256_set1_epi64x(std::bit_cast<int64_t>(window)), shifts), mask);
                        } else {
                            lows = _mm256_set_epi64x(
                                static_cast<int64_t>(read_low<SPLIT_POINT>(low_data, I + 3)),
                                static_cast<int64_t>(read_low<SPLIT_POINT>(low_data, I + 2)),
                                static_cast<int64_t>(read_low<SPLIT_POINT>(low_data, I + 1)),
                                static_cast<int64_t>(read_low<SPLIT_POINT>(low_data, I)));
                        }
                        const __m256i plus = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(plus_positions + I));
                        const __m256i minus = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(minus_positions + I));
                        const __m256i high = _mm256_slli_epi64(_mm256_sub_epi64(plus, minus), SPLIT_POINT);
                        const __m256i values = _mm256_add_epi64(base_vector, _mm256_or_si256(high, lows));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + I), values);
                    }
                }(), ...);
            } else
#endif
            {
                ([&] {
                    const U high = static_cast<U>(plus_positions[I] - minus_positions[I]) << SPLIT_POINT;
                    const U low = static_cast<U>(read_low<SPLIT_POINT>(low_data, I));
                    output[I] = static_cast<T>(base_u + (high | low));
                }(), ...);
            }
        }

        template<uint8_t SPLIT_POINT>
        size_t get_elements_worker(size_t startIndex, size_t count, T* output) const {
            constexpr size_t MAX_BATCH = 256;
            if (count == 0 || startIndex >= m_num_elements) [[unlikely]] return 0;
            count = std::min(count, m_num_elements - startIndex);
            using U = std::make_unsigned_t<T>;
            const U base_u = static_cast<U>(base);
            const uint64_t* low_data = L.data();

            if (h == 0) {
                if constexpr (SPLIT_POINT == 0) {
                    std::fill_n(output, count, base);
                } else {
                    for (size_t i = 0; i < count; ++i)
                        output[i] = static_cast<T>(base_u + static_cast<U>(read_low<SPLIT_POINT>(low_data, startIndex + i)));
                }
                return count;
            }

            const size_t plus_start = startIndex ? G_plus->select0(startIndex) + 1 : 0;
            const size_t minus_start = startIndex ? G_minus->select0(startIndex) + 1 : 0;
            ZeroPositionReader plus(G_plus->raw_data_ptr(), plus_start);
            ZeroPositionReader minus(G_minus->raw_data_ptr(), minus_start);
            alignas(64) size_t plus_positions[MAX_BATCH];
            alignas(64) size_t minus_positions[MAX_BATCH];
            for (size_t produced = 0; produced < count;) {
                const size_t batch = std::min(count - produced, MAX_BATCH);
                plus.read(plus_positions, batch);
                minus.read(minus_positions, batch);
                const auto reconstruct_scalar = [&](size_t i) {
                    U high = 0;
                    if constexpr (SPLIT_POINT < sizeof(T) * 8)
                        high = static_cast<U>(plus_positions[i] - minus_positions[i]) << SPLIT_POINT;
                    const U low = static_cast<U>(read_low<SPLIT_POINT>(low_data, startIndex + produced + i));
                    output[produced + i] = static_cast<T>(base_u + (high | low));
                };
                size_t i = 0;
                if constexpr (SPLIT_POINT > 0 && SPLIT_POINT < sizeof(T) * 8) {
                    const size_t prefix = std::min(batch, (64 - ((startIndex + produced) & 63)) & 63);
                    for (; i < prefix; ++i) reconstruct_scalar(i);
                    for (; batch - i >= 64; i += 64) {
                        const uint64_t* block = low_data + ((startIndex + produced + i) / 64) * SPLIT_POINT;
                        reconstruct_aligned64<SPLIT_POINT>(block, plus_positions + i, minus_positions + i,
                                                           base_u, output + produced + i,
                                                           std::make_index_sequence<64>{});
                    }
                }
                for (; i < batch; ++i) reconstruct_scalar(i);
                produced += batch;
            }
            return count;
        }

    public:
        using IGEF<T>::serialize;
        using IGEF<T>::load;

        ~B_STAR_GEF() override = default;

        // Default constructor
        B_STAR_GEF() : h(0), b(0), m_num_elements(0), base(0) {
        }

        // 2. Copy Constructor
        B_STAR_GEF(const B_STAR_GEF &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              L(other.L),
              h(other.h),
              b(other.b),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            if (other.h > 0) {
                G_plus = std::make_unique<GapBitVectorType>(*other.G_plus);
                if constexpr (RandomAccess) {
                    G_plus->enable_select0();
                }

                G_minus = std::make_unique<GapBitVectorType>(*other.G_minus);
                if constexpr (RandomAccess) {
                    G_minus->enable_select0();
                }
            } else {
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        // Friend swap function for copy-and-swap idiom
        friend void swap(B_STAR_GEF &first, B_STAR_GEF &second) noexcept {
            using std::swap;
            swap(first.L, second.L);
            swap(first.h, second.h);
            swap(first.b, second.b);
            swap(first.m_num_elements, second.m_num_elements);
            swap(first.base, second.base);
            swap(first.G_plus, second.G_plus);
            swap(first.G_minus, second.G_minus);
        }

        // 3. Copy Assignment Operator (using copy-and-swap idiom)
        B_STAR_GEF &operator=(const B_STAR_GEF &other) {
            if (this != &other) {
                B_STAR_GEF temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // 4. Move Constructor
        B_STAR_GEF(B_STAR_GEF &&other) noexcept
            : IGEF<T>(std::move(other)),
              G_plus(std::move(other.G_plus)),
              G_minus(std::move(other.G_minus)),
              L(std::move(other.L)),
              h(other.h),
              b(other.b),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            // Leave the moved-from object in a valid, empty state
            other.h = 0;
            other.m_num_elements = 0;
            other.base = T{};
            other.G_plus = nullptr;
            other.G_minus = nullptr;
        }


        // 5. Move Assignment Operator
        B_STAR_GEF &operator=(B_STAR_GEF &&other) noexcept {
            if (this != &other) {
                G_plus = std::move(other.G_plus);
                G_minus = std::move(other.G_minus);
                L = std::move(other.L);
                h = other.h;
                b = other.b;
                m_num_elements = other.m_num_elements;
                base = other.base;
            }
            return *this;
        }


        template<typename C>
        B_STAR_GEF(const C& S, const BlockParameters<T>& parameters)
            : B_STAR_GEF(S, APPROXIMATE_SPLIT_POINT, nullptr, &parameters) {}

        // Constructor
        template<typename C>
        B_STAR_GEF(const C &S,
                   SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT,
                   CompressionBuildMetrics* metrics = nullptr,
                   const BlockParameters<T>* parameters = nullptr) {
            using clock = std::chrono::steady_clock;
            std::chrono::time_point<clock> split_start;
            if (metrics) {
                split_start = clock::now();
            }
            const size_t N = S.size();
            if (parameters) parameters->validate(S);
            m_num_elements = N;
            if (N == 0) {
                b = 0;
                h = 0;
                base = T{};
                if (metrics) {
                    double split_seconds = std::chrono::duration<double>(clock::now() - split_start).count();
                    metrics->record_partition(split_seconds, 0.0, 0.0, 0, 0, 0);
                }
                return;
            }

            // Independent reductions avoid minmax_element's pairwise branch
            // and let the compiler vectorize contiguous input scans.
            auto minimum = *S.begin();
            auto maximum = minimum;
            for (const auto value : S) {
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
            base = parameters ? parameters->base : minimum;
            const T max_val = maximum;
            
            uint8_t total_bits = value_width(base, max_val);

            GapComputation gap_computation{};

            if (parameters) {
                total_bits = parameters->width;
                b = parameters->split;
                if (b < total_bits) {
                    using U = std::make_unsigned_t<T>;
                    U previous = 0;
                    for (const T value : S) {
                        const U residual = static_cast<U>(value) - static_cast<U>(base);
                        const U current = residual >> b;
                        const size_t gap = current >= previous ? current - previous : previous - current;
                        if (gap > UINT32_MAX) {
                            throw std::overflow_error("GEF unary gap exceeds decoder capacity");
                        }
                        size_t& sum = current >= previous ? gap_computation.sum_of_positive_gaps
                                                         : gap_computation.sum_of_negative_gaps;
                        if (gap > SIZE_MAX - N - sum) {
                            throw std::overflow_error("GEF gap vector is too large");
                        }
                        sum += gap;
                        previous = current;
                    }
                }
            } else switch (strategy) {
                case APPROXIMATE_SPLIT_POINT:
                    std::tie(b, gap_computation) = approximate_optimal_split_point(S, base, max_val);
                    break;
                case OPTIMAL_SPLIT_POINT:
                    std::tie(b, gap_computation) = optimal_split_point(S, base, max_val);
                    break;
                default:
                    throw std::invalid_argument("Invalid split point strategy");
            }
            h = total_bits - b;

            double split_seconds = 0.0;
            if (metrics) {
                split_seconds = std::chrono::duration<double>(clock::now() - split_start).count();
            }

            std::chrono::time_point<clock> allocation_start;
            if (metrics) {
                allocation_start = clock::now();
            }

            if (b > 0) {
                L = sdsl::int_vector<>(N, 0, b);
            } else {
                L = sdsl::int_vector<>(0);
            }

            if (h == 0) {
                double allocation_seconds = 0.0;
                if (metrics) {
                    allocation_seconds = std::chrono::duration<double>(clock::now() - allocation_start).count();
                }

                std::chrono::time_point<clock> population_start;
                if (metrics) {
                    population_start = clock::now();
                }
                
                // Recheck L size if b > 0
                if (b > 0 && L.size() != N) {
                    L = sdsl::int_vector<>(N, 0, b);
                }

                // Special case: no high bits, only L is needed.
                if (b > 0) {
                    for (size_t i = 0; i < N; ++i) {
                        using U = std::make_unsigned_t<T>;
                        L[i] = static_cast<U>(S[i]) - static_cast<U>(base);
                    }
                }
                G_plus = nullptr;
                G_minus = nullptr;
                if (metrics) {
                    double population_seconds = std::chrono::duration<double>(clock::now() - population_start).count();
                    metrics->record_partition(split_seconds,
                                              allocation_seconds,
                                              population_seconds,
                                              N,
                                              0,
                                              b);
                }
                return;
            }

            const size_t g_plus_bits = gap_computation.sum_of_positive_gaps + N;
            const size_t g_minus_bits = gap_computation.sum_of_negative_gaps + N;

            G_plus = std::make_unique<GapBitVectorType>(g_plus_bits);
            G_minus = std::make_unique<GapBitVectorType>(g_minus_bits);
            
            // advise_huge_pages(G_plus->raw_data_ptr(), G_plus->size());
            // advise_huge_pages(G_minus->raw_data_ptr(), G_minus->size());

            double allocation_seconds = 0.0;
            if (metrics) {
                allocation_seconds = std::chrono::duration<double>(clock::now() - allocation_start).count();
            }

            std::chrono::time_point<clock> population_start;
            if (metrics) {
                population_start = clock::now();
            }

            // ========== OPTIMIZED POPULATION PHASE ==========
            // Use direct bit manipulation to eliminate virtual function call overhead (~3N calls)
            // Both SDSL and SUX implementations provide raw_data_ptr() for this optimization
            using U = std::make_unsigned_t<T>;
            const U low_mask = b ? (U(~U(0)) >> (sizeof(T) * 8 - b)) : U(0);
            
            uint64_t* g_plus_data = G_plus->raw_data_ptr();
            uint64_t* g_minus_data = G_minus->raw_data_ptr();

            FastBitWriter<GapBitVectorType::reverse_bit_order> plus_writer(g_plus_data);
            FastBitWriter<GapBitVectorType::reverse_bit_order> minus_writer(g_minus_data);
            U lastHighBits = 0;

            // L writer state
            uint64_t* l_data = L.data();
            uint64_t l_buffer = 0;
            uint8_t l_bits = 0;
            size_t l_word_idx = 0;

            for (size_t i = 0; i < N; ++i) {
                const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
                
                // Store low part
                if (b > 0) [[likely]] {
                    uint64_t val = static_cast<uint64_t>(element_u & low_mask);
                    l_buffer |= (val << l_bits);
                    if (l_bits + b >= 64) {
                        l_data[l_word_idx++] = l_buffer;
                        if (l_bits + b > 64) {
                            l_buffer = val >> (64 - l_bits);
                            l_bits = l_bits + b - 64;
                        } else {
                            l_buffer = 0;
                            l_bits = 0;
                        }
                    } else {
                        l_bits += b;
                    }
                }

                // Compute high part and gap
                const U currentHighBits = element_u >> b;
                const int64_t gap = static_cast<int64_t>(currentHighBits) - static_cast<int64_t>(lastHighBits);

                // Write gap encoding using inline bit operations (no virtual calls)
                if (gap > 0) {
                    plus_writer.set_ones_range(static_cast<uint64_t>(gap));
                } else if (gap < 0) {
                    minus_writer.set_ones_range(static_cast<uint64_t>(-gap));
                }
                // Note: gap == 0 means no ones to write
                
                // Write terminators (always 0)
                minus_writer.skip_zeros(1);
                plus_writer.skip_zeros(1);

                lastHighBits = currentHighBits;
            }
            
            // Flush L buffer
            if (b > 0 && l_bits > 0) {
                l_data[l_word_idx] = l_buffer;
            }
            // ===== END POPULATION PHASE =====

            // Enable select0 support (rank not needed, operator[] only uses select0)
            if constexpr (RandomAccess) {
                G_plus->enable_select0();
                G_minus->enable_select0();
            }

            if (metrics) {
                double population_seconds = std::chrono::duration<double>(clock::now() - population_start).count();
                metrics->record_partition(split_seconds,
                                          allocation_seconds,
                                          population_seconds,
                                          N,
                                          0,
                                          b);
            }
        }

        template <size_t... Is>
        size_t dispatch_worker(size_t b, size_t start, size_t count, T* out, std::index_sequence<Is...>) const {
            using WorkerPtr = size_t (B_STAR_GEF::*)(size_t, size_t, T*) const;

            // Create the table
            static constexpr WorkerPtr table[] = { &B_STAR_GEF::get_elements_worker<Is>... };

            if (b >= sizeof...(Is)) {
                throw std::invalid_argument("Invalid b value");
            }

            // Call the function
            return (this->*table[b])(start, count, out);
        }

        // 2. Your actual function implementation
    private:
        size_t decode_prefix_no_support(size_t count, std::vector<T>& output) const {
            if (count == 0 || m_num_elements == 0) return 0;
            const size_t endIndex = std::min(count, m_num_elements);
            // Caller guarantees output has at least 'count' slots.

            using U = std::make_unsigned_t<T>;
            using Acc = unsigned long long;

            if (h == 0) [[unlikely]] {
                if (b == 0) [[unlikely]] {
                    for (size_t i = 0; i < endIndex; ++i) output[i] = base;
                    return endIndex;
                }
                for (size_t i = 0; i < endIndex; ++i) {
                    const U low = static_cast<U>(L[i]);
                    const Acc sum = static_cast<Acc>(base) + static_cast<Acc>(low);
                    output[i] = static_cast<T>(sum);
                }
                return endIndex;
            }

            FastUnaryDecoder<GapBitVectorType::reverse_bit_order> plus_decoder(G_plus->raw_data_ptr(), G_plus->size(), 0);
            FastUnaryDecoder<GapBitVectorType::reverse_bit_order> minus_decoder(G_minus->raw_data_ptr(), G_minus->size(), 0);

            long long running_high = 0;
            for (size_t i = 0; i < endIndex; ++i) {
                running_high += static_cast<long long>(plus_decoder.next());
                running_high -= static_cast<long long>(minus_decoder.next());

                const U low = (b > 0) ? static_cast<U>(L[i]) : U(0);
                const U high_u = static_cast<U>(running_high);
                const U offset = (b < sizeof(U) * 8) ? (low | (high_u << b)) : low;
                const Acc sum = static_cast<Acc>(base) + static_cast<Acc>(offset);
                output[i] = static_cast<T>(sum);
            }
            return endIndex;
        }

    public:
        size_t get_elements(size_t startIndex, size_t count, std::vector<T>& output) const override {
            if constexpr (RandomAccess) {
                if (count == 0 || startIndex >= m_num_elements) return 0;
                if (output.size() < count)
                    throw std::invalid_argument("output buffer is smaller than requested count");
                return get_elements_into(startIndex, count, output.data());
            } else {
                return ::gef::internal::no_support::get_elements_force_from_zero<T>(
                    startIndex, count, m_num_elements, output,
                    [this](size_t n, std::vector<T>& out) { return this->decode_prefix_no_support(n, out); });
            }
        }

        // Decode directly into caller-owned contiguous storage. The caller must
        // provide space for min(count, size() - startIndex) values when startIndex
        // is valid; this avoids partition-local allocation and copying.
        size_t get_elements_into(size_t startIndex, size_t count, T* output) const
            requires (RandomAccess) {
            return dispatch_worker(b, startIndex, count, output, std::make_index_sequence<65>{});
        }

        [[gnu::always_inline]] inline T operator[](size_t index) const override {
            if constexpr (!RandomAccess) {
                std::vector<T> scratch;
                return ::gef::internal::no_support::at_force_from_zero<T>(
                    index, scratch,
                    [this](size_t n, std::vector<T>& out) { return this->decode_prefix_no_support(n, out); });
            }
            if (h == 0) [[unlikely]]
                return base + (b > 0 ? L[index] : 0);
            
            using U = std::make_unsigned_t<T>;

            // Initiate L access early to hide memory latency while computing select0
            const U low = (b > 0) ? static_cast<U>(L[index]) : U(0);

            const size_t zero_rank = index + 1;
            // Stage the two independent select queries together when the
            // backend supports it; other bitvectors retain their scalar path.
            size_t pos_gaps, neg_gaps;
            if constexpr (requires { G_plus->select0_pair_unchecked(*G_minus, zero_rank); }) {
                const auto positions = G_plus->select0_pair_unchecked(*G_minus, zero_rank);
                pos_gaps = positions.first - zero_rank;
                neg_gaps = positions.second - zero_rank;
            } else {
                pos_gaps = G_plus->select0_unchecked(zero_rank) - zero_rank;
                neg_gaps = G_minus->select0_unchecked(zero_rank) - zero_rank;
            }

            const U high_u = static_cast<U>(
                static_cast<long long>(pos_gaps) - static_cast<long long>(neg_gaps));

            return base + (low | (high_u << b));
        }

        void serialize(std::ofstream &ofs) const override {
            if (!ofs.is_open()) {
                throw std::runtime_error("Could not open file for serialization");
            }
            ofs.write(reinterpret_cast<const char *>(&h), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&b), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&m_num_elements), sizeof(m_num_elements));
            ofs.write(reinterpret_cast<const char *>(&base), sizeof(T));
            
            if (b > 0) {
                L.serialize(ofs);
            }
            
            if (h > 0) {
                G_plus->serialize(ofs);
                G_minus->serialize(ofs);
            }
        }

        void load(std::ifstream &ifs) override {
            ifs.read(reinterpret_cast<char *>(&h), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&b), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&m_num_elements), sizeof(m_num_elements));
            ifs.read(reinterpret_cast<char *>(&base), sizeof(T));
            
            if (b > 0) {
                L.load(ifs);
            } else {
                L = sdsl::int_vector<>(0);
            }
            
            if (h > 0) {
                G_plus = std::make_unique<GapBitVectorType>(GapBitVectorType::load(ifs));
                if constexpr (RandomAccess) {
                    G_plus->enable_select0();
                }

                G_minus = std::make_unique<GapBitVectorType>(GapBitVectorType::load(ifs));
                if constexpr (RandomAccess) {
                    G_minus->enable_select0();
                }
            } else {
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return m_num_elements;
        }

        // Owned C++ object storage and dynamic capacities; allocator metadata
        // and allocation-size-class rounding are intentionally excluded.
        [[nodiscard]] size_t allocated_size_in_bytes() const
            requires requires(const GapBitVectorType& bits) { bits.allocated_size_in_bytes(); } {
            return sizeof(*this) + sdsl_allocated_payload_bytes(L) +
                (G_plus ? G_plus->allocated_size_in_bytes() : 0) +
                (G_minus ? G_minus->allocated_size_in_bytes() : 0);
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

        [[nodiscard]] size_t size_in_bytes_without_supports() const override {
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };
            size_t total_bytes = 0;
            // Raw payload bits only
            total_bytes += sdsl::size_in_bytes(L);
            if (h > 0) {
                total_bytes += bits_to_bytes(G_plus->size());
                total_bytes += bits_to_bytes(G_minus->size());
            }
            // Fixed metadata
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            return total_bytes;
        }

        [[nodiscard]] size_t theoretical_size_in_bytes() const override {
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };
            size_t total_bytes = 0;
            
            // L vector: use width * size formula (theoretical)
            total_bytes += bits_to_bytes(L.size() * L.width());
            
            // G_plus, G_minus bit vectors (if they exist)
            if (h > 0) {
                total_bytes += bits_to_bytes(G_plus->size());
                total_bytes += bits_to_bytes(G_minus->size());
            }
            
            // Fixed metadata
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            
            return total_bytes;
        }

        [[nodiscard]] uint8_t split_point() const override {
            return this->b;
        }
    };
    } // namespace internal
} // namespace gef

#endif
