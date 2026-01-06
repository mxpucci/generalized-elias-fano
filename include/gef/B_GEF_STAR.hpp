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
#include <stdexcept>


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
#include "RLE_GEF.hpp"
#include "gap_computation_utils.hpp"
#include "CompressionProfile.hpp"
#include "FastUnaryDecoder.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVector.hpp"
#include "../datastructures/PastaBitVector.hpp"
#include "FastBitWriter.hpp"


namespace gef {

    template<typename T, typename GapBitVectorType = PastaGapBitVector>
    class B_GEF_STAR : public IGEF<T> {
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

        template<typename C>
        static double approximated_optimal_split_point(const C &S, const T min, const T max) {
            const GapComputation gap_computation = variation_of_original_vec(S, min, max);
            const size_t total_variation = gap_computation.sum_of_negative_gaps + gap_computation.sum_of_positive_gaps;

            return log2(log(2) * total_variation / S.size());
        }

        template<typename C>
        static std::pair<uint8_t, GapComputation> approximate_optimal_split_point(const C &S,
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


        template<typename C>
        static std::pair<uint8_t, GapComputation>
        optimal_split_point(const C &S, const T min, const T max) {
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
        B_GEF_STAR() : h(0), b(0), m_num_elements(0), base(0) {
        }

        // 2. Copy Constructor
        B_GEF_STAR(const B_GEF_STAR &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              L(other.L),
              h(other.h),
              b(other.b),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            if (other.h > 0) {
                G_plus = std::make_unique<GapBitVectorType>(*other.G_plus);
                G_plus->enable_select0();

                G_minus = std::make_unique<GapBitVectorType>(*other.G_minus);
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
            swap(first.m_num_elements, second.m_num_elements);
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
        B_GEF_STAR &operator=(B_GEF_STAR &&other) noexcept {
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


        // Constructor
        template<typename C>
        B_GEF_STAR(const std::shared_ptr<IBitVectorFactory> &bit_vector_factory,
                   const C &S,
                   SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT,
                   CompressionBuildMetrics* metrics = nullptr) {
            using clock = std::chrono::steady_clock;
            std::chrono::time_point<clock> split_start;
            if (metrics) {
                split_start = clock::now();
            }
            const size_t N = S.size();
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
                        L[i] = S[i] - base;
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
                minus_writer.set_zero();
                plus_writer.set_zero();

                lastHighBits = currentHighBits;
            }
            
            // Flush L buffer
            if (b > 0 && l_bits > 0) {
                l_data[l_word_idx] = l_buffer;
            }
            // ===== END POPULATION PHASE =====

            // Enable select0 support (rank not needed, operator[] only uses select0)
            G_plus->enable_select0();
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

        size_t get_elements(size_t startIndex, size_t count, std::vector<T>& output) const override {
            if (count == 0 || startIndex >= size()) return 0;
            
            if (output.size() < count)
                throw std::invalid_argument("output buffer is smaller than requested count");

            const size_t endIndex = std::min(startIndex + count, size());
            const size_t actual_count = endIndex - startIndex;
            T* out_ptr = output.data();

            if (h == 0) [[unlikely]] {
                const uint64_t* l_data = L.data();
                size_t current_bit = startIndex * b;
                
                // Local bit buffer state
                uint64_t bit_buffer = 0;
                size_t bits_in_buffer = 0;
                size_t word_idx = current_bit >> 6;
                size_t bit_offset = current_bit & 63;
                
                // Prime buffer
                if (b > 0) {
                    bit_buffer = l_data[word_idx];
                    bit_buffer >>= bit_offset; // Align
                    bits_in_buffer = 64 - bit_offset;
                }

                const uint64_t l_mask = (b == 64) ? ~0ULL : ((1ULL << b) - 1);

                for (size_t i = 0; i < actual_count; ++i) {
                    if (b == 0) {
                        out_ptr[i] = base; 
                        continue;
                    }

                    uint64_t val;
                    if (bits_in_buffer >= b) {
                        // Happy path: bits are in the buffer
                        val = bit_buffer & l_mask;
                        bit_buffer >>= b;
                        bits_in_buffer -= b;
                    } else {
                        // Refill path: Split the read across words
                        // 1. Take remaining bits from current buffer
                        val = bit_buffer; 
                        
                        // 2. Get the rest from the next word
                        size_t needed = b - bits_in_buffer;
                        word_idx++;
                        uint64_t next_word = l_data[word_idx];
                        
                        // Mask carefully to avoid UB if needed==64
                        uint64_t next_part = next_word & ((needed == 64) ? ~0ULL : ((1ULL << needed) - 1));
                        
                        val |= (next_part << bits_in_buffer);
                        
                        // 3. Update buffer to hold only what's left of next_word
                        bit_buffer = next_word >> needed;
                        bits_in_buffer = 64 - needed;
                    }

                    out_ptr[i] = base + static_cast<T>(val);
                }
                return actual_count;
            }

            // 1. Prefix Sum Calculation (Jump to startIndex)
            size_t pos_prefix = 0;
            size_t neg_prefix = 0;
            size_t plus_start_bit = 0;
            size_t minus_start_bit = 0;

            if (startIndex > 0) {
                // select0(k) returns the position of the k-th zero.
                size_t p_sel = G_plus->select0_unchecked(startIndex);
                size_t n_sel = G_minus->select0_unchecked(startIndex);
                
                pos_prefix = p_sel - (startIndex - 1);
                neg_prefix = n_sel - (startIndex - 1);
                plus_start_bit = p_sel + 1;
                minus_start_bit = n_sel + 1;
            }
            
            // Signed accumulator
            long long current_high = static_cast<long long>(pos_prefix) - static_cast<long long>(neg_prefix);

            // 2. Setup Decoders
            using DecoderType = FastUnaryDecoder<GapBitVectorType::reverse_bit_order>;
            DecoderType plus_decoder(G_plus->raw_data_ptr(), G_plus->size(), std::min(plus_start_bit, G_plus->size()));
            DecoderType minus_decoder(G_minus->raw_data_ptr(), G_minus->size(), std::min(minus_start_bit, G_minus->size()));

            // 3. Setup Low Bit Reader
            const uint64_t* l_data = L.data();
            size_t current_l_bit = startIndex * b;
            size_t l_word_idx = current_l_bit >> 6;
            size_t l_bit_offset = current_l_bit & 63;
            
            uint64_t l_buffer = 0;
            size_t l_bits_available = 0;

            if (b > 0) {
                l_buffer = l_data[l_word_idx];
                l_buffer >>= l_bit_offset;
                l_bits_available = 64 - l_bit_offset;
            }
            const uint64_t l_mask = (b == 64) ? ~0ULL : ((1ULL << b) - 1);

            // 4. Batch Loop
            size_t processed = 0;
            constexpr size_t BATCH_SIZE = 128;
            uint32_t p_gaps[BATCH_SIZE];
            uint32_t n_gaps[BATCH_SIZE];

            while (processed < actual_count) {
                size_t batch_len = std::min(BATCH_SIZE, actual_count - processed);

                // Bulk Decode
                size_t p_fetched = plus_decoder.next_batch(p_gaps, batch_len);
                size_t n_fetched = minus_decoder.next_batch(n_gaps, batch_len);
                
                // Zero-fill remainder (safety)
                for(; p_fetched < batch_len; ++p_fetched) p_gaps[p_fetched] = 0;
                for(; n_fetched < batch_len; ++n_fetched) n_gaps[n_fetched] = 0;

                // Inner Combination Loop
                for (size_t k = 0; k < batch_len; ++k) {
                    current_high += p_gaps[k];
                    current_high -= n_gaps[k];
                    
                    uint64_t low_val = 0;
                    if (b > 0) {
                        if (l_bits_available >= b) {
                            low_val = l_buffer & l_mask;
                            l_buffer >>= b;
                            l_bits_available -= b;
                        } else {
                            // Split read logic
                            low_val = l_buffer; // Take what we have
                            
                            size_t needed = b - l_bits_available;
                            l_word_idx++;
                            uint64_t next_w = l_data[l_word_idx];
                            
                            uint64_t next_part = next_w & ((needed == 64) ? ~0ULL : ((1ULL << needed) - 1));
                            
                            low_val |= (next_part << l_bits_available);
                            
                            l_buffer = next_w >> needed;
                            l_bits_available = 64 - needed;
                        }
                    }

                    using U = std::make_unsigned_t<T>;
                    U high_part = static_cast<U>(current_high);
                    U combined = static_cast<U>(low_val) | (high_part << b);
                    
                    out_ptr[processed + k] = base + static_cast<T>(combined);
                }
                processed += batch_len;
            }

            return actual_count;
        }

        T operator[](size_t index) const override {
            if (h == 0) [[unlikely]]
                return base + L[index];
            
            using U = std::make_unsigned_t<T>;

            // Initiate L access early to hide memory latency while computing select0
            const U low = (b > 0) ? static_cast<U>(L[index]) : U(0);

            const size_t zero_rank = index + 1;
            // Use unchecked select0 - support is guaranteed enabled after construction
            const size_t pos_gaps = G_plus->select0_unchecked(zero_rank) - zero_rank;
            const size_t neg_gaps = G_minus->select0_unchecked(zero_rank) - zero_rank;

            const U high_u = static_cast<U>(
                static_cast<long long>(pos_gaps) - static_cast<long long>(neg_gaps));

            return base + static_cast<T>(low | (high_u << b));
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

        void load(std::ifstream &ifs, const std::shared_ptr<IBitVectorFactory> bit_vector_factory) override {
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
                G_plus->enable_select0();

                G_minus = std::make_unique<GapBitVectorType>(GapBitVectorType::load(ifs));
                G_minus->enable_select0();
            } else {
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return m_num_elements;
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
