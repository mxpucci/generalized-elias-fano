//
// Created by Michelangelo Pucci on 03/08/25.
//

#ifndef B_STAR_GEF_HPP
#define B_STAR_GEF_HPP

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


        template<uint8_t SPLIT_POINT>
        size_t get_elements_worker(size_t startIndex, size_t count, std::vector<T>& output) const {
            constexpr size_t MAX_BATCH = 512;
            
            if (count == 0 || startIndex >= m_num_elements) [[unlikely]] {
                return 0;
            }
            if (output.size() < count) [[unlikely]] {
                throw std::invalid_argument("output buffer is smaller than requested count");
            }

            const size_t endIndex = std::min(startIndex + count, m_num_elements);
            size_t write_index = startIndex;
            size_t produced_count = 0;


            const uint64_t* l_raw_data = nullptr;
            uint64_t l_buffer_curr = 0;
            size_t l_word_idx = 0;
            uint32_t l_bit_offset = 0;

            if constexpr (SPLIT_POINT > 0) {
                l_raw_data = L.data();
                // Calculate initial bit position in L
                const size_t l_bit_global_pos = startIndex * SPLIT_POINT;
                l_word_idx = l_bit_global_pos >> 6;
                l_bit_offset = static_cast<uint32_t>(l_bit_global_pos & 63);
                l_buffer_curr = l_raw_data[l_word_idx];
            }

            // Mask for low bits
            constexpr uint64_t l_mask = (SPLIT_POINT == 64) ? ~0ULL : ((1ULL << SPLIT_POINT) - 1);


            /* --- Fast Path: No High Bits (h=0) --- */
            if (h == 0) [[unlikely]] {
                T* out_ptr = output.data(); 
                
                while (write_index < endIndex) {
                    const size_t remaining = endIndex - write_index;
                    for (size_t i = 0; i < remaining; ++i) {
                        uint64_t l_val = 0;
                        if constexpr (SPLIT_POINT > 0) {
                            // Extract 'b' bits
                            uint64_t word0 = l_buffer_curr >> l_bit_offset;
                            
                            if (l_bit_offset + SPLIT_POINT > 64) {
                                // Crossing boundary
                                l_word_idx++;
                                l_buffer_curr = l_raw_data[l_word_idx]; // Load next
                                uint64_t word1 = l_buffer_curr << (64 - l_bit_offset);
                                l_val = (word0 | word1) & l_mask;
                            } else {
                                l_val = word0 & l_mask;
                                if (l_bit_offset + SPLIT_POINT == 64) {
                                    l_word_idx++;
                                    l_buffer_curr = l_raw_data[l_word_idx]; 
                                }
                            }
                            
                            // Advance offset
                            l_bit_offset = (l_bit_offset + SPLIT_POINT) & 63;
                        }
                        
                        out_ptr[produced_count++] = base + static_cast<T>(l_val);
                    }
                    write_index += remaining;
                }
                return produced_count;
            }

            // --- Standard Path: High Bits Present ---
            
            // 1. Setup High-Part State
            size_t plusGapStart = 0;
            size_t minusGapStart = 0;
            
            using U = std::make_unsigned_t<T>;
            U running_high = 0;

            if (startIndex > 0) [[unlikely]] {
                plusGapStart = G_plus->select0(startIndex) + 1;
                minusGapStart = G_minus->select0(startIndex) + 1;
                const size_t pos_gaps = (plusGapStart - 1) - startIndex;
                const size_t neg_gaps = (minusGapStart - 1) - startIndex;
                running_high = static_cast<U>(static_cast<int64_t>(pos_gaps) - static_cast<int64_t>(neg_gaps));
            }

            // 2. Initialize Decoders
            FastUnaryDecoder<GapBitVectorType::reverse_bit_order> plus_decoder(G_plus->raw_data_ptr(), G_plus->size(), plusGapStart);
            FastUnaryDecoder<GapBitVectorType::reverse_bit_order> minus_decoder(G_minus->raw_data_ptr(), G_minus->size(), minusGapStart);

            uint32_t plus_gaps[MAX_BATCH];
            uint32_t minus_gaps[MAX_BATCH];
            
            T* out_ptr = output.data();

            while (write_index < endIndex) {
                const size_t remaining = endIndex - write_index;
                const size_t batch_size = std::min(remaining, MAX_BATCH);
                
                // Batch Decode Gaps
                plus_decoder.prefetch();
                minus_decoder.prefetch();

                plus_decoder.next_batch(plus_gaps, batch_size);
                minus_decoder.next_batch(minus_gaps, batch_size);

                // Hot Loop: Merge L extraction + H update + Write
                for (size_t i = 0; i < batch_size; ++i) {
                    // --- 1. Extract Low Part (Inline Stream Reader) ---
                    uint64_t l_val = 0;
                    if constexpr (SPLIT_POINT > 0) {
                        uint64_t word0 = l_buffer_curr >> l_bit_offset;
                        
                        if (l_bit_offset + SPLIT_POINT > 64) {
                            l_word_idx++;
                            l_buffer_curr = l_raw_data[l_word_idx]; 
                            uint64_t word1 = l_buffer_curr << (64 - l_bit_offset);
                            l_val = (word0 | word1) & l_mask;
                        } else {
                            l_val = word0 & l_mask;
                            if (l_bit_offset + SPLIT_POINT == 64) {
                                l_word_idx++;
                                l_buffer_curr = l_raw_data[l_word_idx];
                            }
                        }
                        l_bit_offset = (l_bit_offset + SPLIT_POINT) & 63;
                    }

                    // --- 2. Update High Part ---
                    const int32_t delta = static_cast<int32_t>(plus_gaps[i]) - static_cast<int32_t>(minus_gaps[i]);
                    running_high = static_cast<U>(running_high + delta);

                    // --- 3. Fuse and Write ---
                    T high_shifted = 0;
                    if constexpr (SPLIT_POINT < sizeof(T) * 8) {
                         high_shifted = static_cast<T>(running_high << SPLIT_POINT);
                    }
                    const T val = base + static_cast<T>(l_val) + high_shifted;
                    out_ptr[produced_count++] = val;
                }

                write_index += batch_size;
            }

            return produced_count;
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
                G_plus->enable_select0();

                G_minus = std::make_unique<GapBitVectorType>(*other.G_minus);
                G_minus->enable_select0();
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


        // Constructor
        template<typename C>
        B_STAR_GEF(const std::shared_ptr<IBitVectorFactory> &bit_vector_factory,
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

        template <size_t... Is>
        size_t dispatch_worker(size_t b, size_t start, size_t count, std::vector<T>& out, std::index_sequence<Is...>) const {
            using WorkerPtr = size_t (B_STAR_GEF::*)(size_t, size_t, std::vector<T>&) const;

            // Create the table
            static constexpr WorkerPtr table[] = { &B_STAR_GEF::get_elements_worker<Is>... };

            if (b >= sizeof...(Is)) {
                throw std::invalid_argument("Invalid b value");
            }

            // Call the function
            return (this->*table[b])(start, count, out);
        }

        // 2. Your actual function implementation
        size_t get_elements(size_t startIndex, size_t count, std::vector<T>& output) const override {
            // 65 is the number of cases (0 to 64)
            return dispatch_worker(b, startIndex, count, output, std::make_index_sequence<65>{});
        }

        T operator[](size_t index) const override {
            if (h == 0) [[unlikely]]
                return base + (b > 0 ? L[index] : 0);
            
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
