//
// Created by Michelangelo Pucci on 25/07/25.
//

#ifndef B_GEF_HPP
#define B_GEF_HPP

#include "gap_computation_utils.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits>
#include <utility>
#include <chrono>
#include <stdexcept>
#include "IGEF.hpp"
#include "CompressionProfile.hpp"
#include "FastBitWriter.hpp"
#include "FastUnaryDecoder.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVector.hpp"
#include "../datastructures/PastaBitVector.hpp"

#if __has_include(<experimental/simd>) && !defined(GEF_DISABLE_SIMD)
#include <experimental/simd>
namespace stdx = std::experimental;
#define GEF_EXPERIMENTAL_SIMD_ENABLED
#endif

namespace gef {
    template<typename T, typename ExceptionBitVectorType = PastaExceptionBitVector, typename GapBitVectorType = PastaGapBitVector>
    class B_GEF : public IGEF<T> {
    private:
        static uint8_t bits_for_range(const T min_val, const T max_val) {
            using WI = __int128;
            using WU = unsigned __int128;
            const WI min_w = static_cast<WI>(min_val);
            const WI max_w = static_cast<WI>(max_val);
            const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);
            if (range <= 1) return 1;
            size_t bits = 0;
            WU x = range - 1; // bits needed to represent values in [0, range-1]
            while (x > 0) { ++bits; x >>= 1; }
            // Clamp to the size of T to prevent undefined behavior in shifts
            return static_cast<uint8_t>(std::min<size_t>(bits, sizeof(T) * 8));
        }

        // Bit-vector such that B[i] = 0 <==> 0 <= highPart(i) - highPart(i - 1) <= h
        std::unique_ptr<ExceptionBitVectorType> B;

        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that 0 <= highPart(i) - highPart(i - 1) <= h
         */
        std::unique_ptr<GapBitVectorType> G_plus;

        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that 0 <= highPart(i - 1) - highPart(i) <= h
         */
        std::unique_ptr<GapBitVectorType> G_minus;

        // high parts
        sdsl::int_vector<> H;

        // low parts
        sdsl::int_vector<> L;

        // The split point that rules which bits are stored in H and in L
        uint8_t b;
        uint8_t h;
        size_t m_num_elements; // Explicitly store size since L might be empty if b=0

        /**
         * The minimum of the encoded sequence, so that we store the shifted sequence
         * that falls in the range [0, max S - base]
         * This tricks may boost compression and allows us to implicitly store negative numbers
         */
        T base;

        static size_t evaluate_space(const size_t N,
                                     const size_t total_bits,
                                     const uint8_t b,
                                     const GapComputation &gc) {
            if (N == 0)
                return sizeof(T) + 2;  // Just metadata

            if (b >= total_bits) {
                // All in L, no high bits
                // L stores N values, each with total_bits bits
                // But we need to account for sdsl overhead
                // Estimate: at least N * total_bits bits rounded to bytes, plus metadata
                size_t l_bytes = ((N * static_cast<size_t>(total_bits) + 7) / 8);
                // Add metadata overhead
                return l_bytes + sizeof(T) + 2;
            }

            // Exceptions
            const auto exceptions = gc.positive_exceptions_count + gc.negative_exceptions_count;
            if (exceptions == 0 || exceptions > N) {
                // Invalid - fallback
                size_t fallback_bits = N * static_cast<size_t>(total_bits);
                return ((fallback_bits + 7) / 8) + sizeof(T) + 2;
            }
            const size_t non_exc = N - exceptions;

            // Helper to convert bits to bytes
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };

            // Calculate components in bytes (theoretical)
            // B_GEF components: L, H, B, G+, G-
            // L size: N * b bits
            // H size: exceptions * (total_bits - b) bits
            // B size: N bits
            // G+ size: g_plus_bits
            // G- size: g_minus_bits
            
            size_t l_bytes = bits_to_bytes(N * b);
            size_t h_bytes = bits_to_bytes(exceptions * static_cast<size_t>(total_bits - b));
            size_t b_bits = N;
            size_t b_bytes = bits_to_bytes(b_bits);
            size_t g_plus_bits = gc.sum_of_positive_gaps_without_exception + non_exc;
            size_t g_plus_bytes = bits_to_bytes(g_plus_bits);
            size_t g_minus_bits = gc.sum_of_negative_gaps_without_exception + non_exc;
            size_t g_minus_bytes = bits_to_bytes(g_minus_bits);

            // Metadata
            size_t metadata = sizeof(T) + sizeof(h) + sizeof(b);

            return l_bytes + h_bytes + b_bytes + g_plus_bytes + g_minus_bytes + metadata;
        }

        template<typename C>
        static std::pair<uint8_t, GapComputation> approximate_optimal_split_point
        (const C &S,
            const T min,
            const T max) {
            const size_t total_bits = bits_for_range(min, max);

            const auto gc = variation_of_original_vec(S, min, max);
            const auto total_variation = gc.sum_of_negative_gaps + gc.sum_of_positive_gaps;

            const double approximated_split_point = log2(log(2) * total_variation / S.size());
            // Clamp candidate b to [0, total_bits]
            auto clamp_b = [&](double x) -> uint8_t {
                long long bi = static_cast<long long>(std::llround(x));
                long long lo = 0;
                long long hi = static_cast<long long>(total_bits);
                if (bi < lo) bi = lo;
                if (bi > hi) bi = hi;
                return static_cast<uint8_t>(bi);
            };
            uint8_t bFloor = clamp_b(approximated_split_point);
            uint8_t bCeil = clamp_b(approximated_split_point + 1);

            const uint8_t min_b = std::min(bFloor, bCeil);
            const uint8_t max_b = std::max(bFloor, bCeil);

            // Only compute gap computations for the 2 candidates we actually need
            const auto gcs = total_variation_of_shifted_vec_with_multiple_shifts(
                S, min, max, min_b, max_b, ExceptionRule::BGEF);

            size_t best_index = 0;
            size_t best_space = evaluate_space(S.size(), total_bits, min_b, gcs[0]);
            
            // Also consider h=0 (b=total_bits) as a baseline candidate
            size_t h0_space = evaluate_space(S.size(), total_bits, total_bits, GapComputation{});
            
            const auto tmpSpace = evaluate_space(S.size(), total_bits, max_b, gcs[gcs.size() - 1]);
            if (tmpSpace < best_space) {
                best_index = gcs.size() - 1;
                best_space = tmpSpace;
            }
            
            if (h0_space < best_space) {
                 return {static_cast<uint8_t>(total_bits), GapComputation{}};
            }

            return {static_cast<uint8_t>(min_b + best_index), gcs[best_index]};
        }

        template<typename C>
        static std::pair<uint8_t, GapComputation> brute_force_optima_split_point(
            const C &S,
            const T min_val,
            const T max_val) {
            const size_t N = S.size();
            if (N == 0) return {0u, GapComputation{}};

            const uint8_t total_bits = bits_for_range(min_val, max_val);

            // Compute all gap computations in a single pass
            const auto gcs = compute_all_gap_computations(S, min_val, max_val, ExceptionRule::BGEF, total_bits);

            uint8_t best_b = total_bits;
            size_t best_space = evaluate_space(N, total_bits, total_bits, GapComputation{});

            // Check all b values for true optimality
            for (uint8_t b = 1; b < total_bits; ++b) {
                const size_t space = evaluate_space(N, total_bits, b, gcs[b]);
                if (space < best_space) {
                    best_b = b;
                    best_space = space;
                }
            }
            
            // Return the gap computation for the selected split point
            if (best_b < total_bits) {
                return {best_b, gcs[best_b]};
            } else {
                return {total_bits, GapComputation{}};
            }
        }

        static std::make_unsigned_t<T>
        highPart(const std::make_unsigned_t<T> x, const uint8_t total_bits, const uint8_t highBits) {
            const uint8_t lowBits = total_bits - highBits;
            if (lowBits >= sizeof(T) * 8) return 0;
            return static_cast<std::make_unsigned_t<T>>(x >> lowBits);
        }

        static std::make_unsigned_t<T>
        lowPart(const std::make_unsigned_t<T> x, const uint8_t lowBits) {
            if (lowBits >= sizeof(T) * 8) return x;
            const std::make_unsigned_t<T> mask = (static_cast<std::make_unsigned_t<T>>(1) << lowBits) - 1;
            return static_cast<std::make_unsigned_t<T>>(x & mask);
        }

        template<uint8_t SPLIT_POINT>
        size_t get_elements_worker(size_t startIndex, size_t count, std::vector<T>& output) const {
            if (count == 0 || startIndex >= size()) {
                return 0;
            }
            if (output.size() < count) {
                throw std::invalid_argument("output buffer is smaller than requested count");
            }

            const size_t endIndex = std::min(startIndex + count, size());
            size_t write_index = 0;

            using Wide = std::conditional_t<(sizeof(T) < 4), uint32_t, std::make_unsigned_t<T>>;
            using Acc = std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>;
            using U = std::make_unsigned_t<T>;

            // L vector streaming setup
            const uint64_t* l_raw_data = nullptr;
            uint64_t l_buffer_curr = 0;
            size_t l_word_idx = 0;
            uint32_t l_bit_offset = 0;

            if constexpr (SPLIT_POINT > 0) {
                l_raw_data = L.data();
                const size_t l_bit_global_pos = startIndex * SPLIT_POINT;
                l_word_idx = l_bit_global_pos >> 6;
                l_bit_offset = static_cast<uint32_t>(l_bit_global_pos & 63);
                l_buffer_curr = l_raw_data[l_word_idx];
            }
            constexpr uint64_t l_mask = (SPLIT_POINT == 64) ? ~0ULL : ((1ULL << SPLIT_POINT) - 1);

            auto read_L_val = [&]() -> Wide {
                if constexpr (SPLIT_POINT == 0) return 0;
                
                uint64_t l_val = 0;
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
                return static_cast<Wide>(l_val);
            };

            if (h == 0) [[unlikely]] {
                T* out_ptr = output.data();
                for (size_t i = startIndex; i < endIndex; ++i) {
                    Wide low = read_L_val();
                    const Acc sum = static_cast<Acc>(base) + static_cast<Acc>(low);
                    out_ptr[write_index++] = static_cast<T>(sum);
                }
                return write_index;
            }
            
            size_t exception_rank = B->rank_unchecked(startIndex);
            const size_t zero_before = startIndex - exception_rank;
            long long high_signed = (exception_rank == 0) ? 0LL : static_cast<long long>(H[exception_rank - 1]);
            const uint64_t* b_data = B->raw_data_ptr();
            auto read_bit = [b_data](size_t pos) -> bool { return (b_data[pos >> 6] >> (pos & 63)) & 1; };
            const bool start_is_exception = read_bit(startIndex);
            if (!start_is_exception && zero_before > 0) {
                size_t gap_run_start = 0;
                if (exception_rank > 0) {
                    const size_t last_exc_pos = B->select(exception_rank);
                    gap_run_start = last_exc_pos - exception_rank + 1;
                }
                const size_t pos_gap_start = gap_run_start > 0 ? G_plus->select0_unchecked(gap_run_start) - (gap_run_start - 1) : 0;
                const size_t pos_gap_end = G_plus->select0_unchecked(zero_before) - (zero_before - 1);
                const size_t neg_gap_start = gap_run_start > 0 ? G_minus->select0_unchecked(gap_run_start) - (gap_run_start - 1) : 0;
                const size_t neg_gap_end = G_minus->select0_unchecked(zero_before) - (zero_before - 1);
                high_signed += static_cast<long long>(pos_gap_end - pos_gap_start);
                high_signed -= static_cast<long long>(neg_gap_end - neg_gap_start);
            } else if (start_is_exception) {
                ++exception_rank;
                high_signed = static_cast<long long>(H[exception_rank - 1]);
            }
            const size_t plus_bits = G_plus->size();
            const size_t minus_bits = G_minus->size();
            const size_t plus_start_bit = zero_before > 0 ? std::min(G_plus->select0_unchecked(zero_before) + 1, plus_bits) : 0;
            const size_t minus_start_bit = zero_before > 0 ? std::min(G_minus->select0_unchecked(zero_before) + 1, minus_bits) : 0;
            FastUnaryDecoder<GapBitVectorType::reverse_bit_order> plus_decoder(G_plus->raw_data_ptr(), plus_bits, plus_start_bit);
            FastUnaryDecoder<GapBitVectorType::reverse_bit_order> minus_decoder(G_minus->raw_data_ptr(), minus_bits, minus_start_bit);
            
            constexpr size_t GAP_BATCH = 64;
            uint32_t pos_buffer[GAP_BATCH];
            uint32_t neg_buffer[GAP_BATCH];
            size_t pos_size = 0, pos_index = 0;
            size_t neg_size = 0, neg_index = 0;
            
            size_t i = startIndex;
            T* out_ptr = output.data();

            // Loop 1: Alignment to 64
            while (i < endIndex && (i & 63)) {
                if (read_bit(i)) [[unlikely]] {
                    ++exception_rank;
                    high_signed = static_cast<long long>(H[exception_rank - 1]);
                } else [[likely]] {
                    if (pos_index >= pos_size) [[unlikely]] { 
                        pos_size = plus_decoder.next_batch(pos_buffer, GAP_BATCH); 
                        pos_index = 0; 
                        if (pos_size == 0) [[unlikely]] { pos_buffer[0] = static_cast<uint32_t>(plus_decoder.next()); pos_size = 1; } 
                    }
                    if (neg_index >= neg_size) [[unlikely]] { 
                        neg_size = minus_decoder.next_batch(neg_buffer, GAP_BATCH); 
                        neg_index = 0; 
                        if (neg_size == 0) [[unlikely]] { neg_buffer[0] = static_cast<uint32_t>(minus_decoder.next()); neg_size = 1; } 
                    }
                    high_signed += static_cast<long long>(pos_buffer[pos_index++]);
                    high_signed -= static_cast<long long>(neg_buffer[neg_index++]);
                }
                const U high_val = static_cast<U>(high_signed);
                const Wide low = read_L_val();
                Wide high_shifted = 0;
                if constexpr (SPLIT_POINT < sizeof(Wide) * 8) {
                    high_shifted = static_cast<Wide>(high_val) << SPLIT_POINT;
                }
                const Wide offset = low | high_shifted;
                out_ptr[write_index++] = static_cast<T>(static_cast<Acc>(base) + static_cast<Acc>(offset));
                ++i;
            }

            const uint64_t* b_blocks = b_data + (i >> 6);
            while (i + 64 <= endIndex) {
                uint64_t block = *b_blocks++;
                if (block == 0) {
                    // Fast path: 64 non-exceptions
                    int k = 0;
                    for (; k < 64; ++k) {
                         if (pos_index >= pos_size) [[unlikely]] { pos_size = plus_decoder.next_batch(pos_buffer, GAP_BATCH); pos_index = 0; if (pos_size == 0) [[unlikely]] { pos_buffer[0] = static_cast<uint32_t>(plus_decoder.next()); pos_size = 1; } }
                        if (neg_index >= neg_size) [[unlikely]] { neg_size = minus_decoder.next_batch(neg_buffer, GAP_BATCH); neg_index = 0; if (neg_size == 0) [[unlikely]] { neg_buffer[0] = static_cast<uint32_t>(minus_decoder.next()); neg_size = 1; } }
                        high_signed += static_cast<long long>(pos_buffer[pos_index++]);
                        high_signed -= static_cast<long long>(neg_buffer[neg_index++]);
                        U high_val = static_cast<U>(high_signed);
                        Wide low = read_L_val();
                        Wide high_shifted = 0;
                        if constexpr (SPLIT_POINT < sizeof(Wide) * 8) {
                            high_shifted = static_cast<Wide>(high_val) << SPLIT_POINT;
                        }
                        const Wide offset = low | high_shifted;
                        out_ptr[write_index++] = static_cast<T>(static_cast<Acc>(base) + static_cast<Acc>(offset));
                    }
                } else { 
                     for (int k = 0; k < 64; ++k) {
                        if ((block >> k) & 1) { ++exception_rank; high_signed = static_cast<long long>(H[exception_rank - 1]); }
                        else {
                             if (pos_index >= pos_size) [[unlikely]] { pos_size = plus_decoder.next_batch(pos_buffer, GAP_BATCH); pos_index = 0; if (pos_size == 0) [[unlikely]] { pos_buffer[0] = static_cast<uint32_t>(plus_decoder.next()); pos_size = 1; } }
                            if (neg_index >= neg_size) [[unlikely]] { neg_size = minus_decoder.next_batch(neg_buffer, GAP_BATCH); neg_index = 0; if (neg_size == 0) [[unlikely]] { neg_buffer[0] = static_cast<uint32_t>(minus_decoder.next()); neg_size = 1; } }
                            high_signed += static_cast<long long>(pos_buffer[pos_index++]); high_signed -= static_cast<long long>(neg_buffer[neg_index++]);
                        }
                        U high_val = static_cast<U>(high_signed);
                        Wide low = read_L_val();
                        Wide high_shifted = 0;
                        if constexpr (SPLIT_POINT < sizeof(Wide) * 8) {
                            high_shifted = static_cast<Wide>(high_val) << SPLIT_POINT;
                        }
                        const Wide offset = low | high_shifted;
                        out_ptr[write_index++] = static_cast<T>(static_cast<Acc>(base) + static_cast<Acc>(offset));
                     }
                }
                i += 64;
            }
            while (i < endIndex) {
                if (read_bit(i)) [[unlikely]] { ++exception_rank; high_signed = static_cast<long long>(H[exception_rank - 1]); }
                else [[likely]] {
                    if (pos_index >= pos_size) [[unlikely]] { pos_size = plus_decoder.next_batch(pos_buffer, GAP_BATCH); pos_index = 0; if (pos_size == 0) [[unlikely]] { pos_buffer[0] = static_cast<uint32_t>(plus_decoder.next()); pos_size = 1; } }
                    if (neg_index >= neg_size) [[unlikely]] { neg_size = minus_decoder.next_batch(neg_buffer, GAP_BATCH); neg_index = 0; if (neg_size == 0) [[unlikely]] { neg_buffer[0] = static_cast<uint32_t>(minus_decoder.next()); neg_size = 1; } }
                    high_signed += static_cast<long long>(pos_buffer[pos_index++]); high_signed -= static_cast<long long>(neg_buffer[neg_index++]);
                }
                const U high_val = static_cast<U>(high_signed);
                const Wide low = read_L_val();
                Wide high_shifted = 0;
                if constexpr (SPLIT_POINT < sizeof(Wide) * 8) {
                    high_shifted = static_cast<Wide>(high_val) << SPLIT_POINT;
                }
                const Wide offset = low | high_shifted;
                out_ptr[write_index++] = static_cast<T>(static_cast<Acc>(base) + static_cast<Acc>(offset));
                ++i;
            }
            return write_index;
        }

        template <size_t... Is>
        size_t dispatch_worker(size_t b, size_t start, size_t count, std::vector<T>& out, std::index_sequence<Is...>) const {
            using WorkerPtr = size_t (B_GEF::*)(size_t, size_t, std::vector<T>&) const;
            static constexpr WorkerPtr table[] = { &B_GEF::get_elements_worker<Is>... };
            if (b >= sizeof...(Is)) {
                throw std::invalid_argument("Invalid b value");
            }
            return (this->*table[b])(start, count, out);
        }

    public:
        using IGEF<T>::serialize;
        using IGEF<T>::load;

        ~B_GEF() override = default;

        // Default constructor
        B_GEF() : h(0), b(0), m_num_elements(0), base(0) {
        }

        // 2. Copy Constructor
        B_GEF(const B_GEF &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              H(other.H),
              L(other.L),
              h(other.h),
              b(other.b),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            if (other.h > 0) {
                B = std::make_unique<ExceptionBitVectorType>(*other.B);
                B->enable_rank();


                G_plus = std::make_unique<GapBitVectorType>(*other.G_plus);
                G_plus->enable_rank();
                G_plus->enable_select0();

                G_minus = std::make_unique<GapBitVectorType>(*other.G_minus);
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
            swap(first.m_num_elements, second.m_num_elements);
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
              m_num_elements(other.m_num_elements),
              base(other.base) {
            other.h = 0;
            other.m_num_elements = 0;
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
                m_num_elements = other.m_num_elements;
                base = other.base;
            }
            return *this;
        }

        // Constructor
        template<typename C>
        B_GEF(const std::shared_ptr<IBitVectorFactory> &bit_vector_factory,
            const C &S,
            SplitPointStrategy strategy = OPTIMAL_SPLIT_POINT,
            CompressionBuildMetrics* metrics = nullptr) {
            // [Constructor implementation omitted for brevity, same as before]
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
              B = nullptr;
              G_plus = nullptr;
              G_minus = nullptr;
              H.resize(0);
              if (metrics) {
                  double split_seconds = std::chrono::duration<double>(clock::now() - split_start).count();
                  metrics->record_partition(split_seconds, 0.0, 0.0, 0, 0, 0);
              }
              return;
          }

          auto [min_it, max_it] = std::minmax_element(S.begin(), S.end());
          base = *min_it;
          const T max_val = *max_it;

          const uint8_t total_bits = bits_for_range(base, max_val);
          GapComputation gc{};
          if (strategy == OPTIMAL_SPLIT_POINT) {
              std::tie(b, gc) = brute_force_optima_split_point(S, base, max_val);
          } else {
              std::tie(b, gc) = approximate_optimal_split_point(S, base, max_val);
          }
          h = (b >= total_bits) ? 0 : static_cast<uint8_t>(total_bits - b);

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

              using U = std::make_unsigned_t<T>;
              if (b > 0 && L.size() != N) {
                  L = sdsl::int_vector<>(N, 0, b);
              }
              
              if (b > 0) {
                  for (size_t i = 0; i < N; ++i) {
                      L[i] = static_cast<typename sdsl::int_vector<>::value_type>(
                          static_cast<U>(S[i]) - static_cast<U>(base)
                      );
                  }
              }
              B = nullptr;
              G_plus = nullptr;
              G_minus = nullptr;
              H.resize(0);

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

          const size_t exceptions = gc.positive_exceptions_count + gc.negative_exceptions_count;
          const size_t non_exceptions = N - exceptions;
          const size_t g_plus_bits = gc.sum_of_positive_gaps_without_exception + non_exceptions;
          const size_t g_minus_bits = gc.sum_of_negative_gaps_without_exception + non_exceptions;

          B = std::make_unique<ExceptionBitVectorType>(N);
          H = sdsl::int_vector<>(exceptions, 0, h);
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

          size_t h_idx = 0;
          using U = std::make_unsigned_t<T>;
          U lastHighBits = 0;
          
          const uint64_t hbits_u64 = static_cast<uint64_t>(h);
          const U low_mask = (b > 0) ? ((U(1) << b) - 1) : 0;

          uint64_t* b_data = B->raw_data_ptr();
          FastBitWriter<ExceptionBitVectorType::reverse_bit_order> b_writer(b_data);

          uint64_t* g_plus_data = G_plus->raw_data_ptr();
          uint64_t* g_minus_data = G_minus->raw_data_ptr();
          FastBitWriter<GapBitVectorType::reverse_bit_order> plus_writer(g_plus_data);
          FastBitWriter<GapBitVectorType::reverse_bit_order> minus_writer(g_minus_data);

          if (N > 0) {
              const U element_u = static_cast<U>(S[0]) - static_cast<U>(base);
              if (b > 0) {
                  L[0] = static_cast<typename sdsl::int_vector<>::value_type>(element_u & low_mask);
              }
              const U current_high_part = element_u >> b;
              
              b_writer.set_ones_range(1);
              H[h_idx++] = current_high_part;
              lastHighBits = current_high_part;
          }

          for (size_t i = 1; i < N; ++i) {
              const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
              if (b > 0) {
                  L[i] = static_cast<typename sdsl::int_vector<>::value_type>(element_u & low_mask);
              }
              
              const U current_high_part = element_u >> b;
              using WI = __int128;
              const WI gap = static_cast<WI>(current_high_part) - static_cast<WI>(lastHighBits);
              const uint64_t abs_gap = (gap >= 0) ? static_cast<uint64_t>(gap) : static_cast<uint64_t>(-gap);
              
              const bool is_exception = ((static_cast<unsigned __int128>(abs_gap) + 2) > static_cast<unsigned __int128>(hbits_u64));

              if (is_exception) [[unlikely]] {
                  b_writer.set_ones_range(1);
                  H[h_idx++] = current_high_part;
              } else [[likely]] {
                  b_writer.set_zero();
                  if (gap >= 0) {
                      plus_writer.set_ones_range(static_cast<uint64_t>(abs_gap));
                  } else {
                      minus_writer.set_ones_range(static_cast<uint64_t>(abs_gap));
                  }
                  minus_writer.set_zero();
                  plus_writer.set_zero();
              }
              lastHighBits = current_high_part;
          }

          assert(b_writer.position() == N);
          assert(minus_writer.position() == g_minus_bits);
          assert(plus_writer.position() == g_plus_bits);

          B->enable_rank();

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
                                        exceptions,
                                        b);
          }
      }

        size_t get_elements(size_t startIndex, size_t count, std::vector<T>& output) const override {
            return dispatch_worker(b, startIndex, count, output, std::make_index_sequence<65>{});
        }

        T operator[](size_t index) const override {
            // Case 1: No high bits are used (h=0).
            // The value is fully stored in the L vector.
            if (h == 0) [[unlikely]] {
                using Acc = std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>;
                using Wide = std::conditional_t<(sizeof(T) < 4), uint32_t, std::make_unsigned_t<T>>;
                
                Wide low = (b > 0) ? static_cast<Wide>(L[index]) : Wide(0);

                const Acc sum = static_cast<Acc>(base) + static_cast<Acc>(low);
                return static_cast<T>(sum);
            }

            using Wide = std::conditional_t<(sizeof(T) < 4), uint32_t, std::make_unsigned_t<T>>;
            using Acc = std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>;
            using U = std::make_unsigned_t<T>;
            
            // Initiate L access early to hide memory latency
            Wide low = (b > 0) ? static_cast<Wide>(L[index]) : Wide(0);

            // Helper to compute cumulative gaps using unchecked select0
            const auto cumulative_gaps = [](const std::unique_ptr<GapBitVectorType> &bv, size_t count) -> size_t {
                return (count == 0) ? 0 : bv->select0_unchecked(count) - (count - 1);
            };

            // Check if this position is an exception
            // Optimization: Use raw bit access for B
            const uint64_t* b_data = B->raw_data_ptr();
            bool is_exception = (b_data[index >> 6] >> (index & 63)) & 1;

            if (is_exception) [[unlikely]] {
                const size_t exception_rank = B->rank_unchecked(index + 1);
                const U high_val = static_cast<U>(H[exception_rank - 1]);
                const Wide high_shifted = static_cast<Wide>(high_val) << b;
                const Wide offset = low | high_shifted;
                const Acc sum = static_cast<Acc>(base) + static_cast<Acc>(offset);
                return static_cast<T>(sum);
            }

            // Non-exception case: reconstruct from gaps
            const size_t exception_rank = B->rank_unchecked(index);
            const size_t zero_before = index - exception_rank;
            
            // Start from last exception high (or 0 if no prior exceptions)
            long long high_signed = (exception_rank == 0) ? 0LL : static_cast<long long>(H[exception_rank - 1]);

            // Find where the current gap run starts (after last exception)
            size_t gap_run_start = 0;
            if (exception_rank > 0) {
                const size_t last_exc_pos = B->select_unchecked(exception_rank);
                gap_run_start = last_exc_pos - exception_rank + 1;
            }

            // Add cumulative gaps from gap_run_start to zero_before (inclusive)
            // Using zero_before+1 includes the gap for the element at index
            const size_t pos_gap_end = cumulative_gaps(G_plus, zero_before + 1);
            const size_t pos_gap_start = cumulative_gaps(G_plus, gap_run_start);
            const size_t neg_gap_end = cumulative_gaps(G_minus, zero_before + 1);
            const size_t neg_gap_start = cumulative_gaps(G_minus, gap_run_start);

            high_signed += static_cast<long long>(pos_gap_end - pos_gap_start);
            high_signed -= static_cast<long long>(neg_gap_end - neg_gap_start);

            const U high_val = static_cast<U>(high_signed);
            const Wide high_shifted = static_cast<Wide>(high_val) << b;
            const Wide offset = low | high_shifted;
            const Acc sum = static_cast<Acc>(base) + static_cast<Acc>(offset);
            return static_cast<T>(sum);
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
            ifs.read(reinterpret_cast<char *>(&m_num_elements), sizeof(m_num_elements));
            ifs.read(reinterpret_cast<char *>(&base), sizeof(T));
            
            if (b > 0) {
                L.load(ifs);
            } else {
                L = sdsl::int_vector<>(0);
            }
            
            H.load(ifs);
            if (h > 0) {
                B = std::make_unique<ExceptionBitVectorType>(ExceptionBitVectorType::load(ifs));
                B->enable_rank();

                G_plus = std::make_unique<GapBitVectorType>(GapBitVectorType::load(ifs));
                G_plus->enable_rank();
                G_plus->enable_select0();

                G_minus = std::make_unique<GapBitVectorType>(GapBitVectorType::load(ifs));
                G_minus->enable_rank();
                G_minus->enable_select0();
            } else {
                B = nullptr;
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return m_num_elements;
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

        [[nodiscard]] size_t size_in_bytes_without_supports() const override {
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };
            size_t total_bytes = 0;
            // Raw payload bits only
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sdsl::size_in_bytes(H);
            if (B) {
                total_bytes += bits_to_bytes(B->size());
            }
            if (G_plus) {
                total_bytes += bits_to_bytes(G_plus->size());
            }
            if (G_minus) {
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
            
            // H vector: use width * size formula (theoretical)
            total_bytes += bits_to_bytes(H.size() * H.width());
            
            // B, G_plus, G_minus bit vectors (if they exist)
            if (B) {
                total_bytes += bits_to_bytes(B->size());
            }
            if (G_plus) {
                total_bytes += bits_to_bytes(G_plus->size());
            }
            if (G_minus) {
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
