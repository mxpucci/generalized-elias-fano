//
// Created by Michelangelo Pucci on 03/08/25.
//

#ifndef B_GEF_NO_RLE_NO_RLE_HPP
#define B_GEF_NO_RLE_NO_RLE_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits>
#include <chrono>

// SIMD intrinsics for optimized bit operations
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
#include <immintrin.h>
#elif defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
#include <arm_neon.h>
#endif

#include "IGEF.hpp"
#include "RLE_GEF.hpp"
#include "gap_computation_utils.hpp"
#include "CompressionProfile.hpp"
#include "FastUnaryDecoder.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"
#include "FastBitWriter.hpp"


namespace gef {

    template<typename T>
    class B_GEF_STAR : public IGEF<T> {
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

        static size_t evaluate_space(const GapComputation &gap_computation,
                                     const uint8_t b,
                                     const size_t total_bits) {
            const size_t N = gap_computation.negative_gaps + gap_computation.positive_gaps;
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

        static double approximated_optimal_split_point(const std::vector<T> &S, const T min, const T max) {
            const GapComputation gap_computation = variation_of_original_vec(S, min, max);
            const size_t total_variation = gap_computation.sum_of_negative_gaps + gap_computation.sum_of_positive_gaps;

            return log2(log(2) * total_variation / S.size());
        }

        static std::pair<uint8_t, GapComputation> approximate_optimal_split_point(const std::vector<T> &S,
            const T min, const T max) {
            using WI = __int128;
            using WU = unsigned __int128;
            const WI min_w = static_cast<WI>(min);
            const WI max_w = static_cast<WI>(max);
            const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);

            size_t total_bits = 1;
            if (range > 1) {
                WU x = range - 1;
                total_bits = 0;
                while (x > 0) {
                    ++total_bits;
                    x >>= 1;
                }
            }
            total_bits = std::min<size_t>(total_bits, sizeof(T) * 8);

            const double approx_b = approximated_optimal_split_point(S, min, max);
            
            if (ceil(approx_b) == floor(approx_b)) {
                const uint8_t b_clamped = static_cast<uint8_t>(
                    std::max(0.0, std::min<double>(total_bits, floor(approx_b))));
                return {
                    b_clamped,
                    variation_of_shifted_vec(S, min, max, b_clamped, ExceptionRule::None)
                };
            }

            const double ceil_candidate = std::min<double>(total_bits, ceil(approx_b));
            const double floor_candidate = std::max(0.0, floor(approx_b));
            const uint8_t ceilB = static_cast<uint8_t>(std::min<double>((double)total_bits, ceil_candidate));
            const uint8_t floorB = static_cast<uint8_t>(std::max(0.0, floor_candidate));

            const std::vector<GapComputation> gap_computation =
                    total_variation_of_shifted_vec_with_multiple_shifts(S, min, max, floorB, ceilB, ExceptionRule::None);
            
            size_t best_index = 0;
            size_t best_space = SIZE_MAX;
            for (size_t i = 0; i < gap_computation.size(); i++) {
                const size_t space = evaluate_space(gap_computation[i],
                                                    static_cast<uint8_t>(floorB + i),
                                                    total_bits);
                if (space < best_space) {
                    best_space = space;
                    best_index = i;
                }
            }

            const auto total_bits_stats =
                variation_of_shifted_vec(S,
                                         min,
                                         max,
                                         static_cast<uint8_t>(total_bits),
                                         ExceptionRule::None);
            const size_t total_bits_space =
                evaluate_space(total_bits_stats, static_cast<uint8_t>(total_bits), total_bits);

            if (total_bits_space < best_space) {
                return {static_cast<uint8_t>(total_bits), total_bits_stats};
            }

            return {static_cast<uint8_t>(floorB + best_index), gap_computation[best_index]};
        }


        static std::pair<uint8_t, GapComputation>
        optimal_split_point(const std::vector<T> &S, const T min, const T max) {
            using WI = __int128;
            using WU = unsigned __int128;
            const WI min_w = static_cast<WI>(min);
            const WI max_w = static_cast<WI>(max);
            const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);

            size_t total_bits = 1;
            if (range > 1) {
                WU x = range - 1;
                total_bits = 0;
                while (x > 0) {
                    ++total_bits;
                    x >>= 1;
                }
            }
            total_bits = std::min<size_t>(total_bits, sizeof(T) * 8);

            const double approx_b = approximated_optimal_split_point(S, min, max);
            const size_t min_b = static_cast<size_t>(std::max(0.0, floor(approx_b) - 1));
            const size_t max_b = static_cast<size_t>(std::min(static_cast<double>(total_bits), ceil(approx_b) + 3));

            const std::vector<GapComputation> gap_computation =
                total_variation_of_shifted_vec_with_multiple_shifts(S,
                                                                    min,
                                                                    max,
                                                                    static_cast<uint8_t>(min_b),
                                                                    static_cast<uint8_t>(max_b),
                                                                    ExceptionRule::None);

            size_t best_index = 0;
            size_t best_space = SIZE_MAX;
            for (size_t i = 0; i < gap_computation.size(); i++) {
                const uint8_t current_b = static_cast<uint8_t>(min_b + i);
                const size_t space = evaluate_space(gap_computation[i], current_b, total_bits);
                if (space < best_space) {
                    best_space = space;
                    best_index = i;
                }
            }

            const auto total_bits_stats =
                variation_of_shifted_vec(S,
                                         min,
                                         max,
                                         static_cast<uint8_t>(total_bits),
                                         ExceptionRule::None);
            const size_t total_bits_space =
                evaluate_space(total_bits_stats, static_cast<uint8_t>(total_bits), total_bits);

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

    public:
        using IGEF<T>::serialize;
        using IGEF<T>::load;

        ~B_GEF_STAR() override = default;

        // Default constructor
        B_GEF_STAR() : h(0), b(0), base(0) {
        }

        // 2. Copy Constructor
        B_GEF_STAR(const B_GEF_STAR &other)
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
        friend void swap(B_GEF_STAR &first, B_GEF_STAR &second) noexcept {
            using std::swap;
            swap(first.L, second.L);
            swap(first.h, second.h);
            swap(first.b, second.b);
            swap(first.base, second.base);
            swap(first.G_plus, second.G_plus);
            swap(first.G_minus, second.G_minus);
        }

        // 3. Copy Assignment Operator (using copy-and-swap idiom)
        B_GEF_STAR &operator=(const B_GEF_STAR &other) {
            if (this != &other) {
                B_GEF_STAR temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // 4. Move Constructor
        B_GEF_STAR(B_GEF_STAR &&other) noexcept
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
        B_GEF_STAR &operator=(B_GEF_STAR &&other) noexcept {
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
        B_GEF_STAR(const std::shared_ptr<IBitVectorFactory> &bit_vector_factory,
                   const std::vector<T> &S,
                   SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT,
                   CompressionBuildMetrics* metrics = nullptr) {
            using clock = std::chrono::steady_clock;
            std::chrono::time_point<clock> split_start;
            if (metrics) {
                split_start = clock::now();
            }
            const size_t N = S.size();
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

            auto [min_it, max_it] = std::minmax_element(S.begin(), S.end());
            base = *min_it;
            const T max_val = *max_it;
            
            // Use 128-bit arithmetic to avoid overflow
            using WI = __int128;
            using WU = unsigned __int128;
            const WI min_w = static_cast<WI>(base);
            const WI max_w = static_cast<WI>(max_val);
            const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);
            
            // Calculate total_bits safely, clamped to sizeof(T)*8
            uint8_t total_bits;
            if (range <= 1) {
                total_bits = 1;
            } else {
                size_t bits = 0;
                WU x = range - 1;
                while (x > 0) { ++bits; x >>= 1; }
                total_bits = std::min<size_t>(bits, sizeof(T) * 8);
            }
            
            GapComputation gap_computation;

            switch (strategy) {
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

            L = sdsl::int_vector<>(N, 0, b);
            if (h == 0) {
                double allocation_seconds = 0.0;
                if (metrics) {
                    allocation_seconds = std::chrono::duration<double>(clock::now() - allocation_start).count();
                }

                std::chrono::time_point<clock> population_start;
                if (metrics) {
                    population_start = clock::now();
                }
                // Special case: no high bits, only L is needed.
                for (size_t i = 0; i < N; ++i) {
                    L[i] = S[i] - base;
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

            G_plus = bit_vector_factory->create(g_plus_bits);
            G_minus = bit_vector_factory->create(g_minus_bits);

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
            
            FastBitWriter plus_writer(g_plus_data);
            FastBitWriter minus_writer(g_minus_data);
            U lastHighBits = 0;

            for (size_t i = 0; i < N; ++i) {
                const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
                
                // Store low part
                if (b > 0) [[likely]] {
                    L[i] = static_cast<typename sdsl::int_vector<>::value_type>(element_u & low_mask);
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
                minus_writer.set_zero();
                plus_writer.set_zero();

                lastHighBits = currentHighBits;
            }
            // ===== END POPULATION PHASE =====

            // Enable rank/select support
            G_plus->enable_rank();
            G_plus->enable_select0();
            G_minus->enable_rank();
            G_minus->enable_select0();

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

        std::vector<T> get_elements(size_t startIndex, size_t count) const override {
            std::vector<T> result;
            result.reserve(count);
            
            if (count == 0 || startIndex >= size()) {
                return result;
            }
            
            const size_t endIndex = std::min(startIndex + count, size());
            
            // Fast path: h == 0, all data in L
            if (h == 0) {
                for (size_t i = startIndex; i < endIndex; ++i) {
                    result.push_back(base + L[i]);
                }
                return result;
            }
            
            using U = std::make_unsigned_t<T>;

            const size_t plus_bits = G_plus->size();
            const size_t minus_bits = G_minus->size();
            const size_t pos_prefix =
                (startIndex > 0) ? G_plus->rank(G_plus->select0(startIndex)) : 0;
            const size_t neg_prefix =
                (startIndex > 0) ? G_minus->rank(G_minus->select0(startIndex)) : 0;

            size_t current_pos_sum = pos_prefix;
            size_t current_neg_sum = neg_prefix;

            const size_t plus_start_bit =
                startIndex > 0 ? std::min(G_plus->select0(startIndex) + 1, plus_bits) : 0;
            const size_t minus_start_bit =
                startIndex > 0 ? std::min(G_minus->select0(startIndex) + 1, minus_bits) : 0;

            FastUnaryDecoder plus_decoder(G_plus->raw_data_ptr(), plus_bits, plus_start_bit);
            FastUnaryDecoder minus_decoder(G_minus->raw_data_ptr(), minus_bits, minus_start_bit);

            constexpr size_t GAP_BATCH = 64;
            uint32_t pos_buffer[GAP_BATCH];
            uint32_t neg_buffer[GAP_BATCH];
            size_t pos_size = 0, pos_index = 0;
            size_t neg_size = 0, neg_index = 0;

            // Maintain high as signed throughout to avoid conversions
            long long high_signed = static_cast<long long>(pos_prefix) - static_cast<long long>(neg_prefix);

            for (size_t i = startIndex; i < endIndex; ++i) {
                // Inline gap fetches to reduce function call overhead
                if (pos_index >= pos_size) [[unlikely]] {
                    pos_size = plus_decoder.next_batch(pos_buffer, GAP_BATCH);
                    pos_index = 0;
                    if (pos_size == 0) [[unlikely]] {
                        pos_buffer[0] = static_cast<uint32_t>(plus_decoder.next());
                        pos_size = 1;
                    }
                }
                if (neg_index >= neg_size) [[unlikely]] {
                    neg_size = minus_decoder.next_batch(neg_buffer, GAP_BATCH);
                    neg_index = 0;
                    if (neg_size == 0) [[unlikely]] {
                        neg_buffer[0] = static_cast<uint32_t>(minus_decoder.next());
                        neg_size = 1;
                    }
                }
                high_signed += static_cast<long long>(pos_buffer[pos_index++]);
                high_signed -= static_cast<long long>(neg_buffer[neg_index++]);
                
                const U high_u = static_cast<U>(high_signed);
                const U combined = static_cast<U>(L[i]) | (high_u << b);
                result.push_back(base + static_cast<T>(combined));
            }
            
            return result;
        }

        T operator[](size_t index) const override {
            if (h == 0) [[likely]]
                return base + L[index];
            
            using U = std::make_unsigned_t<T>;

            const size_t plus_bits = G_plus->size();
            const size_t minus_bits = G_minus->size();
            size_t pos_prefix = (index > 0) ? G_plus->rank(G_plus->select0(index)) : 0;
            size_t neg_prefix = (index > 0) ? G_minus->rank(G_minus->select0(index)) : 0;

            const size_t plus_start_bit =
                index > 0 ? std::min(G_plus->select0(index) + 1, plus_bits) : 0;
            const size_t minus_start_bit =
                index > 0 ? std::min(G_minus->select0(index) + 1, minus_bits) : 0;
            FastUnaryDecoder plus_decoder(G_plus->raw_data_ptr(), plus_bits, plus_start_bit);
            FastUnaryDecoder minus_decoder(G_minus->raw_data_ptr(), minus_bits, minus_start_bit);

            uint32_t pos_gap = 0;
            uint32_t neg_gap = 0;
            if (plus_decoder.next_batch(&pos_gap, 1) == 0) [[unlikely]] {
                pos_gap = static_cast<uint32_t>(plus_decoder.next());
            }
            if (minus_decoder.next_batch(&neg_gap, 1) == 0) [[unlikely]] {
                neg_gap = static_cast<uint32_t>(minus_decoder.next());
            }

            pos_prefix += pos_gap;
            neg_prefix += neg_gap;

            const long long high_signed =
                static_cast<long long>(pos_prefix) - static_cast<long long>(neg_prefix);
            const U high_u = static_cast<U>(high_signed);
            const U combined = static_cast<U>(L[index]) | (high_u << b);
            return base + static_cast<T>(combined);
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
} // namespace gef

#endif
