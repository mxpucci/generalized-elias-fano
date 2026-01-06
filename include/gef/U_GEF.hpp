//
// Created by Michelangelo Pucci on 06/07/25.
//

#ifndef U_GEF_HPP
#define U_GEF_HPP

#include "gap_computation_utils.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <string>
#include <vector>
#include <type_traits> // Required for std::make_unsigned
#include <stdexcept>
#include <tuple>
#include <utility>
#include "IGEF.hpp"
#include "RLE_GEF.hpp"
#include "FastBitWriter.hpp"
#include "FastUnaryDecoder.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/SDSLBitVector.hpp"
#include "../datastructures/PastaBitVector.hpp"

#if (defined(__AVX2__) || defined(__SSE4_2__)) && !defined(GEF_DISABLE_SIMD)
#include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
#include <arm_neon.h>
#endif

#if __has_include(<experimental/simd>) && !defined(GEF_DISABLE_SIMD)
#include <experimental/simd>
namespace stdx = std::experimental;
#define GEF_EXPERIMENTAL_SIMD_ENABLED
#endif

namespace gef {
    namespace internal {
    template<typename T, typename ExceptionBitVectorType = PastaExceptionBitVector, typename GapBitVectorType = PastaGapBitVector>
    class U_GEF : public IGEF<T> {
    private:
        static uint8_t bits_for_range(const T min_val, const T max_val) {
            using WI = __int128;
            using WU = unsigned __int128;
            const WI min_w = static_cast<WI>(min_val);
            const WI max_w = static_cast<WI>(max_val);
            const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);
            if (range <= 1) return 1;
            size_t bits = 0;
            WU x = range - 1;
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
        std::unique_ptr<GapBitVectorType> G;

        // high parts
        sdsl::int_vector<> H;

        // low parts
        sdsl::int_vector<> L;

        // The split point that rules which bits are stored in H and in L
        uint8_t b;
        uint8_t h;
        bool reversed;
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
                                     const GapComputation &gc,
                                     bool reversed) {
            if (N == 0)
                return sizeof(T) + 3;  // Just metadata (added reversed)

            if (b >= total_bits) {
                // All in L, no high bits
                size_t l_bytes = ((N * static_cast<size_t>(total_bits) + 7) / 8);
                return l_bytes + sizeof(T) + 3;
            }

            // Exceptions calculation based on tracking mode
            // In Normal U_GEF, all negative gaps are exceptions.
            // In Reversed U_GEF, all positive gaps are exceptions (as they become negative).
            // 'positive_exceptions_count' and 'negative_exceptions_count' in gc now track MAGNITUDE exceptions only.
            
            size_t exceptions;
            size_t g_bits_sum;
            
            if (reversed) {
                // Reversed Mode:
                // - All strictly positive gaps are exceptions (they become negative after reversal).
                // - Zero gaps remain non-exceptions.
                const size_t strictly_positive_gaps =
                    (gc.positive_gaps > gc.zero_gaps) ? (gc.positive_gaps - gc.zero_gaps) : 0;
                exceptions = strictly_positive_gaps + gc.negative_exceptions_count;
                
                // Gap Sum = Sum of Small Negative Gaps (magnitudes)
                g_bits_sum = gc.sum_of_negative_gaps_without_exception;
            } else {
                // Normal Mode:
                // - All negative gaps are exceptions.
                // - Positive gaps encoded. Exception if too large.
                
                // Exceptions = Large Positive Gaps + All Negative Gaps
                exceptions = gc.positive_exceptions_count + gc.negative_gaps;
                
                // Gap Sum = Sum of Small Positive Gaps
                g_bits_sum = gc.sum_of_positive_gaps_without_exception;
            }

            if (exceptions == 0 || exceptions > N) {
                // Invalid - fallback
                size_t fallback_bits = N * static_cast<size_t>(total_bits);
                return ((fallback_bits + 7) / 8) + sizeof(T) + 3;
            }
            const size_t non_exc = N - exceptions;

            // Helper to convert bits to bytes
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };

            // Calculate components in bytes (theoretical)
            // In U_GEF, if h > 0, we have B and G bitvectors.
            // B has size N bits.
            // G stores gaps for non-exceptions.
            // H stores exceptions.
            // L stores N * b bits.
            
            size_t l_bytes = bits_to_bytes(N * b);
            size_t h_bytes = bits_to_bytes(exceptions * static_cast<size_t>(total_bits - b));
            size_t b_bits = N;
            size_t b_bytes = bits_to_bytes(b_bits);
            
            size_t g_bits = g_bits_sum + non_exc;
            
            size_t g_bytes = bits_to_bytes(g_bits);

            // Metadata
            size_t metadata = sizeof(T) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(bool);

            return l_bytes + h_bytes + b_bytes + g_bytes + metadata;
        }

        template<typename C>
        static std::tuple<uint8_t, GapComputation, bool> approximate_optimal_split_point
        (const C &S,
            const T min,
            const T max) {
            const size_t total_bits = bits_for_range(min, max);

            const auto gc = variation_of_original_vec(S, min, max);
            const auto total_variation = gc.sum_of_positive_gaps;

            const double approximated_split_point = log2(total_variation / S.size());
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
                S, min, max, min_b, max_b, ExceptionRule::UGEF);

            size_t best_index = 0;
            bool best_rev = false;
            // Evaluate normal and reversed
            size_t best_space = evaluate_space(S.size(), total_bits, min_b, gcs[0], false);
            size_t rev_space = evaluate_space(S.size(), total_bits, min_b, gcs[0], true);
            if (rev_space < best_space) {
                best_space = rev_space;
                best_rev = true;
            }
            
            // Also consider h=0 (b=total_bits) as a baseline candidate
            size_t h0_space = evaluate_space(S.size(), total_bits, total_bits, GapComputation{}, false); // reversed doesn't matter for h=0

            if (h0_space < best_space) {
                 return {static_cast<uint8_t>(total_bits), GapComputation{}, false};
            }

            // Check max_b
            const auto tmpSpace = evaluate_space(S.size(), total_bits, max_b, gcs[gcs.size() - 1], false);
            if (tmpSpace < best_space) {
                best_index = gcs.size() - 1;
                best_space = tmpSpace;
                best_rev = false;
            }
            const auto tmpSpaceRev = evaluate_space(S.size(), total_bits, max_b, gcs[gcs.size() - 1], true);
            if (tmpSpaceRev < best_space) {
                best_index = gcs.size() - 1;
                best_space = tmpSpaceRev;
                best_rev = true;
            }
            
            // Explicitly check against h=0 case if not covered
            if (h0_space < best_space) {
                 return {static_cast<uint8_t>(total_bits), GapComputation{}, false};
            }

            return {static_cast<uint8_t>(min_b + best_index), gcs[best_index], best_rev};
        }

        template<typename C>
        static std::tuple<uint8_t, GapComputation, bool> brute_force_optima_split_point(
            const C &S,
            const T min_val,
            const T max_val) {
            const size_t N = S.size();
            if (N == 0) return {0u, GapComputation{}, false};

            const uint8_t total_bits = bits_for_range(min_val, max_val);

            // Compute all gap computations in a single pass
            const auto gcs = compute_all_gap_computations(S, min_val, max_val, ExceptionRule::UGEF, total_bits);

            uint8_t best_b = total_bits;
            bool best_rev = false;
            size_t best_space = evaluate_space(N, total_bits, total_bits, GapComputation{}, false);

            // Check all b values for true optimality
            for (uint8_t b = 1; b < total_bits; ++b) {
                const size_t space = evaluate_space(N, total_bits, b, gcs[b], false);
                if (space < best_space) {
                    best_b = b;
                    best_space = space;
                    best_rev = false;
                }
                const size_t space_rev = evaluate_space(N, total_bits, b, gcs[b], true);
                if (space_rev < best_space) {
                    best_b = b;
                    best_space = space_rev;
                    best_rev = true;
                }
            }
            
            // Return the gap computation for the selected split point
            if (best_b < total_bits) {
                return {best_b, gcs[best_b], best_rev};
            } else {
                return {total_bits, GapComputation{}, false};
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

    public:
        using IGEF<T>::serialize;
        using IGEF<T>::load;

        ~U_GEF() override = default;

        // Default constructor
        U_GEF() : h(0), b(0), reversed(false), m_num_elements(0), base(0) {
        }

        // 2. Copy Constructor
        U_GEF(const U_GEF &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              H(other.H),
              L(other.L),
              h(other.h),
              b(other.b),
              reversed(other.reversed),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            if (other.h > 0) {
                B = std::make_unique<ExceptionBitVectorType>(*other.B);
                B->enable_rank();

                G = std::make_unique<GapBitVectorType>(*other.G);
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
            swap(first.reversed, second.reversed);
            swap(first.m_num_elements, second.m_num_elements);
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
              reversed(other.reversed),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            // Leave the moved-from object in a valid, empty state
            other.h = 0;
            other.m_num_elements = 0;
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
                reversed = other.reversed;
                m_num_elements = other.m_num_elements;
                base = other.base;
            }
            return *this;
        }

        // Constructor
        template<typename C>
        U_GEF(const C &S,
              SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT) {
            // [Constructor unchanged]
            const size_t N = S.size();
            m_num_elements = N;
            if (N == 0) {
                b = 0;
                h = 0;
                reversed = false;
                base = T{};
                B = nullptr;
                return;
            }

            auto [min_it, max_it] = std::minmax_element(S.begin(), S.end());
            base = *min_it;
            const T max_val = *max_it;

            const uint8_t total_bits = bits_for_range(base, max_val);
            GapComputation gc{};
            if (strategy == OPTIMAL_SPLIT_POINT) {
                std::tie(b, gc, reversed) = brute_force_optima_split_point(S, base, max_val);
            } else {
                std::tie(b, gc, reversed) = approximate_optimal_split_point(S, base, max_val);
            }
            h = (b >= total_bits) ? 0 : static_cast<uint8_t>(total_bits - b);

            if (b > 0) {
                L = sdsl::int_vector<>(N, 0, b);
            } else {
                // Leave L empty if b=0 to avoid 64-bit width default
                L = sdsl::int_vector<>(0);
            }

            if (h == 0) {
                // All in L - use unsigned arithmetic for efficiency
                using U = std::make_unsigned_t<T>;
                // Re-allocate L if it was skipped above (though b >= total_bits > 0 usually)
                // If total_bits=0 (all elements same), b=0, h=0. L empty is correct.
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
                G = nullptr;
                H.resize(0);
                return;
            }

            // Use precomputed gc for allocation
            size_t exceptions;
            size_t g_bits_sum;
            if (reversed) {
                const size_t strictly_positive_gaps =
                    (gc.positive_gaps > gc.zero_gaps) ? (gc.positive_gaps - gc.zero_gaps) : 0;
                exceptions = strictly_positive_gaps + gc.negative_exceptions_count;
                g_bits_sum = gc.sum_of_negative_gaps_without_exception;
            } else {
                exceptions = gc.positive_exceptions_count + gc.negative_gaps;
                g_bits_sum = gc.sum_of_positive_gaps_without_exception; 
            }
            const size_t non_exceptions = N - exceptions;
            const size_t g_bits = g_bits_sum + non_exceptions;

            B = std::make_unique<ExceptionBitVectorType>(N);
            H = sdsl::int_vector<>(exceptions, 0, h);
            G = std::make_unique<GapBitVectorType>(g_bits);

            // Single pass to populate all structures
            size_t h_idx = 0;
            using U = std::make_unsigned_t<T>;
            U lastHighBits = 0;
            
            // Precompute low mask once - b < total_bits is guaranteed by h > 0
            const U low_mask = (b > 0) ? ((U(1) << b) - 1) : 0;

            uint64_t* b_data = B->raw_data_ptr();
            FastBitWriter<ExceptionBitVectorType::reverse_bit_order> b_writer(b_data);

            uint64_t* g_data = G->raw_data_ptr();
            FastBitWriter<GapBitVectorType::reverse_bit_order> g_writer(g_data);

            // Handle first element separately to avoid i==0 check in loop
            if (N > 0) {
                const U element_u = static_cast<U>(S[0]) - static_cast<U>(base);
                if (b > 0) {
                    L[0] = static_cast<typename sdsl::int_vector<>::value_type>(element_u & low_mask);
                }
                const U current_high_part = element_u >> b;
                
                // i=0 is always an exception
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
                WI gap = static_cast<WI>(current_high_part) - static_cast<WI>(lastHighBits);
                if (reversed) gap = -gap;
                
                // Exception check: gap<0 or gap>=h
                // Even if reversed, gap is now the "effective gap" we want to encode.
                // If gap < 0, we can't encode in G (unary requires >= 0). Exception.
                // If gap >= h, too large. Exception.
                const bool is_exception = (gap < 0) || (static_cast<unsigned __int128>(gap) >= static_cast<unsigned __int128>(h));

                if (is_exception) [[unlikely]] {
                    b_writer.set_ones_range(1);
                    H[h_idx++] = current_high_part;
                } else [[likely]] {
                    b_writer.set_zero();
                    g_writer.set_ones_range(static_cast<uint64_t>(gap));
                    g_writer.set_zero();
                }
                lastHighBits = current_high_part;
            }

            assert(b_writer.position() == N);
            assert(g_writer.position() == g_bits);

            // Enable rank/select support
            B->enable_rank();
            G->enable_rank();
            G->enable_select0();
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
            const T base_value = base;
            
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

            auto read_L_val = [&]() -> U {
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
                return static_cast<U>(l_val);
            };

            // Fast path: h == 0, all data in L
            if (h == 0) [[unlikely]] {
                for (size_t i = startIndex; i < endIndex; ++i) {
                    U low = read_L_val();
                    output[write_index++] = base_value + static_cast<T>(low);
                }
                return write_index;
            }
            
            const uint64_t* b_data = B->raw_data_ptr();
            auto read_bit = [b_data](size_t pos) -> bool {
                return (b_data[pos >> 6] >> (pos & 63)) & 1;
            };

            // --- INITIALIZATION START ---
            size_t exception_rank = B->rank_unchecked(startIndex);
            const size_t zeros_before = startIndex - exception_rank;
            size_t decoder_bit_pos = 0;
            if (zeros_before > 0) {
                decoder_bit_pos = G->select0_unchecked(zeros_before) + 1;
            }
            if (decoder_bit_pos > G->size()) decoder_bit_pos = G->size();
            FastUnaryDecoder<GapBitVectorType::reverse_bit_order> gap_decoder(G->raw_data_ptr(), G->size(), decoder_bit_pos);
            U current_high = 0;
            if (startIndex > 0) {
                current_high = static_cast<U>(H[exception_rank - 1]);
                if (zeros_before > 0) {
                    size_t last_exc_index = B->select(exception_rank);
                    size_t zeros_at_last_exc = (last_exc_index + 1) - exception_rank;
                    if (zeros_before > zeros_at_last_exc) {
                        size_t range_start_bit = (zeros_at_last_exc == 0) ? 0 : (G->select0_unchecked(zeros_at_last_exc) + 1);
                        size_t range_end_bit = decoder_bit_pos;
                        size_t total_bits_in_range = range_end_bit - range_start_bit;
                        size_t num_gaps = zeros_before - zeros_at_last_exc;
                        
                        U sum_gaps = static_cast<U>(total_bits_in_range - num_gaps);
                        if (reversed) current_high -= sum_gaps;
                        else current_high += sum_gaps;
                    }
                }
            }
            
            constexpr size_t GAP_BATCH = 64;
            uint32_t gap_buffer[GAP_BATCH];
            size_t buffer_size = 0;
            size_t buffer_index = 0;

            size_t i = startIndex;
            while (i < endIndex && (i & 63)) {
                 if (read_bit(i)) [[unlikely]] {
                    ++exception_rank;
                    current_high = static_cast<U>(H[exception_rank - 1]);
                } else [[likely]] {
                    if (buffer_index >= buffer_size) [[unlikely]] {
                        buffer_size = gap_decoder.next_batch(gap_buffer, GAP_BATCH);
                        buffer_index = 0;
                        if (buffer_size == 0) [[unlikely]] {
                            gap_buffer[0] = static_cast<uint32_t>(gap_decoder.next());
                            buffer_size = 1;
                        }
                    }
                    if (reversed) current_high -= static_cast<U>(gap_buffer[buffer_index++]);
                    else current_high += static_cast<U>(gap_buffer[buffer_index++]);
                }
                const U low = read_L_val();
                U high_shifted = 0;
                if constexpr (SPLIT_POINT < sizeof(U) * 8) {
                    high_shifted = current_high << SPLIT_POINT;
                }
                output[write_index++] = base_value + static_cast<T>(low | high_shifted);
                ++i;
            }

            const uint64_t* b_blocks = b_data + (i >> 6);
            
            while (i + 64 <= endIndex) {
                uint64_t block = *b_blocks++;
                if (block == 0) { // Fast path: 64 non-exceptions
                    int k = 0;
                    for (; k < 64; ++k) {
                         if (buffer_index >= buffer_size) [[unlikely]] {
                            buffer_size = gap_decoder.next_batch(gap_buffer, GAP_BATCH);
                            buffer_index = 0;
                            if (buffer_size == 0) [[unlikely]] {
                                gap_buffer[0] = static_cast<uint32_t>(gap_decoder.next());
                                buffer_size = 1;
                            }
                        }
                        if (reversed) current_high -= static_cast<U>(gap_buffer[buffer_index++]);
                        else current_high += static_cast<U>(gap_buffer[buffer_index++]);
                        
                        U low = read_L_val();
                        U high_shifted = 0;
                if constexpr (SPLIT_POINT < sizeof(U) * 8) {
                    high_shifted = current_high << SPLIT_POINT;
                }
                output[write_index++] = base_value + static_cast<T>(low | high_shifted);
                    }
                } else { // Slow path: mixed exceptions
                     for (int k = 0; k < 64; ++k) {
                        if ((block >> k) & 1) {
                            ++exception_rank;
                            current_high = static_cast<U>(H[exception_rank - 1]);
                        } else {
                             if (buffer_index >= buffer_size) [[unlikely]] {
                                buffer_size = gap_decoder.next_batch(gap_buffer, GAP_BATCH);
                                buffer_index = 0;
                                if (buffer_size == 0) [[unlikely]] {
                                    gap_buffer[0] = static_cast<uint32_t>(gap_decoder.next());
                                    buffer_size = 1;
                                }
                            }
                            if (reversed) current_high -= static_cast<U>(gap_buffer[buffer_index++]);
                            else current_high += static_cast<U>(gap_buffer[buffer_index++]);
                        }
                        
                        U low = read_L_val();
                        U high_shifted = 0;
                if constexpr (SPLIT_POINT < sizeof(U) * 8) {
                    high_shifted = current_high << SPLIT_POINT;
                }
                output[write_index++] = base_value + static_cast<T>(low | high_shifted);
                     }
                }
                i += 64;
            }

            while (i < endIndex) {
                 if (read_bit(i)) [[unlikely]] {
                    ++exception_rank;
                    current_high = static_cast<U>(H[exception_rank - 1]);
                } else [[likely]] {
                    if (buffer_index >= buffer_size) [[unlikely]] {
                        buffer_size = gap_decoder.next_batch(gap_buffer, GAP_BATCH);
                        buffer_index = 0;
                        if (buffer_size == 0) [[unlikely]] {
                             gap_buffer[0] = static_cast<uint32_t>(gap_decoder.next());
                             buffer_size = 1;
                        }
                    }
                    if (reversed) current_high -= static_cast<U>(gap_buffer[buffer_index++]);
                    else current_high += static_cast<U>(gap_buffer[buffer_index++]);
                }
                const U low = read_L_val();
                U high_shifted = 0;
                if constexpr (SPLIT_POINT < sizeof(U) * 8) {
                    high_shifted = current_high << SPLIT_POINT;
                }
                output[write_index++] = base_value + static_cast<T>(low | high_shifted);
                ++i;
            }

            return write_index;
        }

        template <size_t... Is>
        size_t dispatch_worker(size_t b, size_t start, size_t count, std::vector<T>& out, std::index_sequence<Is...>) const {
            using WorkerPtr = size_t (U_GEF::*)(size_t, size_t, std::vector<T>&) const;
            static constexpr WorkerPtr table[] = { &U_GEF::get_elements_worker<Is>... };
            if (b >= sizeof...(Is)) {
                throw std::invalid_argument("Invalid b value");
            }
            return (this->*table[b])(start, count, out);
        }

        size_t get_elements(size_t startIndex, size_t count, std::vector<T>& output) const override {
            return dispatch_worker(b, startIndex, count, output, std::make_index_sequence<65>{});
        }

        T operator[](size_t index) const override {
            // Case 1: No high bits are used (h=0).
            // All information is stored in the L vector. Reconstruction is trivial.
            if (h == 0) [[unlikely]] {
                if (b == 0) [[unlikely]] return base;
                return base + L[index];
            }

            using U = std::make_unsigned_t<T>;
            // Initiate L access early to hide memory latency
            const U low = (b > 0) ? static_cast<U>(L[index]) : U(0);

            // Helper to compute cumulative gaps: sum of first 'count' gaps
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
                return base + static_cast<T>(low | (high_val << b));
            }

            // Non-exception case: reconstruct from gaps
            const size_t exception_rank = B->rank_unchecked(index);
            const size_t zero_before = index - exception_rank;
            
            // Start from last exception high (or 0 if no prior exceptions)
            U current_high = (exception_rank == 0) ? U(0) : static_cast<U>(H[exception_rank - 1]);
            
            // Find where the current gap run starts (after last exception)
            size_t gap_run_start = 0;
            if (exception_rank > 0) {
                const size_t last_exc_pos = B->select(exception_rank);
                gap_run_start = last_exc_pos - exception_rank + 1;
            }

            // Add cumulative gaps from gap_run_start to zero_before (inclusive)
            // Using zero_before+1 includes the gap for the element at index
            const size_t gap_end = cumulative_gaps(G, zero_before + 1);
            const size_t gap_start = cumulative_gaps(G, gap_run_start);
            
            if (reversed) current_high -= static_cast<U>(gap_end - gap_start);
            else current_high += static_cast<U>(gap_end - gap_start);

            return base + static_cast<T>(low | (current_high << b));
        }

        void serialize(std::ofstream &ofs) const override {
            if (!ofs.is_open()) {
                throw std::runtime_error("Could not open file for serialization");
            }
            ofs.write(reinterpret_cast<const char *>(&h), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&b), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&reversed), sizeof(bool));
            ofs.write(reinterpret_cast<const char *>(&m_num_elements), sizeof(m_num_elements));
            ofs.write(reinterpret_cast<const char *>(&base), sizeof(T));
            
            if (b > 0) {
                L.serialize(ofs);
            }
            H.serialize(ofs);
            if (h > 0) {
                B->serialize(ofs);
                G->serialize(ofs);
            }
        }

        void load(std::ifstream &ifs) override {
            ifs.read(reinterpret_cast<char *>(&h), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&b), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&reversed), sizeof(bool));
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

                G = std::make_unique<GapBitVectorType>(GapBitVectorType::load(ifs));
                G->enable_rank();
                G->enable_select0();
            } else {
                B = nullptr;
                G = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return m_num_elements;
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
            total_bytes += sizeof(reversed);
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
            if (G) {
                total_bytes += bits_to_bytes(G->size());
            }
            // Fixed metadata
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            total_bytes += sizeof(reversed);
            return total_bytes;
        }

        [[nodiscard]] size_t theoretical_size_in_bytes() const override {
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };
            size_t total_bytes = 0;
            
            // L vector: use width * size formula (theoretical)
            size_t l_sz = bits_to_bytes(L.size() * L.width());
            total_bytes += l_sz;
            
            // H vector: use width * size formula (theoretical)
            size_t h_sz = bits_to_bytes(H.size() * H.width());
            total_bytes += h_sz;
            
            // B, G bit vectors (if they exist)
            size_t b_sz = 0;
            size_t g_sz = 0;
            if (B) {
                b_sz = bits_to_bytes(B->size());
                total_bytes += b_sz;
            }
            if (G) {
                g_sz = bits_to_bytes(G->size());
                total_bytes += g_sz;
            }
            
            // Fixed metadata
            size_t meta = sizeof(base) + sizeof(h) + sizeof(b) + sizeof(reversed);
            total_bytes += meta;
            
            return total_bytes;
        }

        [[nodiscard]] uint8_t split_point() const override {
            return this->b;
        }
    };
    } // namespace internal
} // namespace gef

#endif
