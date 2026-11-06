#ifndef RLE_GEF_H
#define RLE_GEF_H

#include <algorithm>
#include <array>
#include <bit>
#include <iostream>
#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits> // Required for std::make_unsigned
#include <utility>
#include <stdexcept>
#include "IGEF.hpp"
#include "AllocatedSize.hpp"
#include "BlockParameters.hpp"
#include "FastBitWriter.hpp"
#include "NoRandomAccess.hpp"

#include "../datastructures/IBitVector.hpp"
#include "../datastructures/SDSLBitVector.hpp"
#include "../datastructures/PastaBitVector.hpp"

#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
#include <immintrin.h>
#endif

#if __has_include(<experimental/simd>) && !defined(GEF_DISABLE_SIMD)
#include <experimental/simd>
namespace stdx = std::experimental;
#define GEF_EXPERIMENTAL_SIMD_ENABLED
#endif

namespace gef {
    namespace internal {
    template<typename T, typename ExceptionBitVectorType = PastaRankBitVector, bool RandomAccess = true>
    class RLE_GEF : public IGEF<T> {
    public:
        // Bit-vector such that B[i] = 1 <==> highPart(i) != highPart(i - 1)
        std::unique_ptr<ExceptionBitVectorType> B;

        // high parts
        sdsl::int_vector<> H;

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

        /**
         * The longest common prefix of two integers x and y represented on total_bits
         */
        static uint8_t LCP(const T x, const T y, const uint8_t total_bits) {
            if (x == y)
                return total_bits;

            // Ensure bitwise operations are performed on unsigned types
            using UnsignedT = std::make_unsigned_t<T>;
            const UnsignedT ux = static_cast<UnsignedT>(x);
            const UnsignedT uy = static_cast<UnsignedT>(y);

            const UnsignedT diff = ux ^ uy;
            int leading_zeros;

            if constexpr (sizeof(T) == 8) {
                // uint64_t
                leading_zeros = __builtin_clzll(diff);
            } else if constexpr (sizeof(T) == 4) {
                // uint32_t
                leading_zeros = __builtin_clz(diff);
            } else if constexpr (sizeof(T) == 2) {
                // uint16_t
                leading_zeros = __builtin_clz(diff) - (sizeof(unsigned int) * 8 - 16);
            } else if constexpr (sizeof(T) == 1) {
                // uint8_t
                leading_zeros = __builtin_clz(diff) - (sizeof(unsigned int) * 8 - 8);
            } else {
                // Fallback for types not explicitly handled, though this should ideally be unreachable
                // for standard integral types. This might indicate a need for a more generic
                // __builtin_clz equivalent or a different approach for arbitrary T.
                // For now, we'll assume it's a power of 2 size and calculate based on that.
                leading_zeros = (8 * sizeof(T) - total_bits);
            }

            return leading_zeros - (8 * sizeof(T) - total_bits);
        }

        template<typename C>
        static uint8_t optimal_split_point(const C& S, const uint8_t total_bits, const T min,
                                           size_t* selected_runs = nullptr) {
            using U = std::make_unsigned_t<T>;
            std::array<size_t, sizeof(T) * 8 + 1> lcp_frequencies{};
            for (size_t i = 1; i < S.size(); i++) {
                const uint8_t lcp = LCP(static_cast<T>(static_cast<U>(S[i]) - static_cast<U>(min)),
                                        static_cast<T>(static_cast<U>(S[i - 1]) - static_cast<U>(min)),
                                        total_bits);
                ++lcp_frequencies[lcp];
            }

            size_t best_b = total_bits;
            size_t best_space = total_bits * S.size();
            size_t best_runs = 0;
            size_t rank_h = 1; // B[0] marks the first run
            for (uint8_t h_val = 1; h_val <= total_bits; h_val++) {
                const uint8_t b = total_bits - h_val;
                rank_h += lcp_frequencies[h_val - 1];
                const size_t space = S.size() * (b + 1) + rank_h * h_val;
                if (space < best_space) {
                    best_space = space;
                    best_b = b;
                    best_runs = rank_h;
                }
            }
            if (selected_runs) *selected_runs = best_runs;
            return best_b;
        }

        static T highPart(const T x, const uint8_t total_bits, const uint8_t highBits) {
            const uint8_t lowBits = total_bits - highBits;
            // Cast to unsigned to ensure logical right shift
            return static_cast<T>(static_cast<std::make_unsigned_t<T>>(x) >> lowBits);
        }

        static T lowPart(const T x, const uint8_t lowBits) {
            if (lowBits >= sizeof(T) * 8) {
                return x;
            }
            // Cast to unsigned to ensure predictable bitwise AND
            const std::make_unsigned_t<T> mask = (static_cast<std::make_unsigned_t<T>>(1) << lowBits) - 1;
            return static_cast<T>(static_cast<std::make_unsigned_t<T>>(x) & mask);
        }

    public:
        using IGEF<T>::serialize;
        using IGEF<T>::load;

        ~RLE_GEF() override = default;

        // Default constructor
        RLE_GEF() : h(0), b(0), m_num_elements(0), base(0) {
        }

        // 2. Copy Constructor
        RLE_GEF(const RLE_GEF &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              H(other.H),
              L(other.L),
              h(other.h),
              b(other.b),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            if (other.h > 0) {
                B = std::make_unique<ExceptionBitVectorType>(*other.B);
                if constexpr (RandomAccess) {
                    B->enable_rank();
                }

            } else {
                B = nullptr;
            }
        }

        // Friend swap function for copy-and-swap idiom
        friend void swap(RLE_GEF &first, RLE_GEF &second) noexcept {
            using std::swap;
            swap(first.B, second.B);
            swap(first.H, second.H);
            swap(first.L, second.L);
            swap(first.h, second.h);
            swap(first.b, second.b);
            swap(first.m_num_elements, second.m_num_elements);
            swap(first.base, second.base);
        }

        // 3. Copy Assignment Operator (using copy-and-swap idiom)
        RLE_GEF &operator=(const RLE_GEF &other) {
            if (this != &other) {
                RLE_GEF temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // 4. Move Constructor
        RLE_GEF(RLE_GEF &&other) noexcept
            : IGEF<T>(std::move(other)),
              B(std::move(other.B)),
              H(std::move(other.H)),
              L(std::move(other.L)),
              h(other.h),
              b(other.b),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            // Leave the moved-from object in a valid, empty state
            other.h = 0;
            other.m_num_elements = 0;
            other.base = T{};
        }


        // 5. Move Assignment Operator
        RLE_GEF &operator=(RLE_GEF &&other) noexcept {
            if (this != &other) {
                B = std::move(other.B);
                H = std::move(other.H);
                L = std::move(other.L);
                h = other.h;
                b = other.b;
                m_num_elements = other.m_num_elements;
                base = other.base;
            }
            return *this;
        }


        template<typename C>
        RLE_GEF(const C& S, const BlockParameters<T>& parameters)
            : RLE_GEF(S, &parameters) {}

        // Constructor
        template<typename C>
        RLE_GEF(const C &S, const BlockParameters<T>* parameters = nullptr) {
            const size_t N = S.size();
            if (parameters) parameters->validate(S);
            m_num_elements = N;
            if (S.size() == 0) {
                b = 0;
                h = 0;
                base = T{};
                B = nullptr;
                return;
            }

            using U = std::make_unsigned_t<T>;
            uint8_t total_bits;
            size_t h_count = 0;
            if (parameters) {
                base = parameters->base;
                total_bits = parameters->width;
                b = parameters->split;
                if (b < total_bits) {
                    h_count = 1;
                    U previous = (static_cast<U>(S[0]) - static_cast<U>(base)) >> b;
                    for (size_t i = 1; i < N; ++i) {
                        const U high = (static_cast<U>(S[i]) - static_cast<U>(base)) >> b;
                        h_count += high != previous;
                        previous = high;
                    }
                }
            } else {
                const auto [min_it, max_it] = std::minmax_element(S.begin(), S.end());
                base = *min_it;
                // Integer width avoids rounding at powers of two and overflow of
                // max-min+1. A constant block needs only its stored base.
                const U range = static_cast<U>(*max_it) - static_cast<U>(base);
                total_bits = static_cast<uint8_t>(std::bit_width(range));
                b = optimal_split_point(S, total_bits, base, &h_count);
            }
            h = total_bits - b;

            if (b > 0) {
                L = sdsl::int_vector<>(S.size(), 0, b);
            } else {
                L = sdsl::int_vector<>(0);
            }

            if (h == 0) {
                // All in L - use unsigned arithmetic for efficiency
                using U = std::make_unsigned_t<T>;
                
                if (b > 0 && L.size() != S.size()) {
                    L = sdsl::int_vector<>(S.size(), 0, b);
                }

                if (b > 0) {
                    for (size_t i = 0; i < S.size(); i++) {
                        L[i] = static_cast<typename sdsl::int_vector<>::value_type>(
                            static_cast<U>(S[i]) - static_cast<U>(base)
                        );
                    }
                }
                B = nullptr;
                H.resize(0);
                return;
            }

            // The split histogram already gives the number of runs. Allocate
            // H exactly once and fill H, L and B in the same source scan.
            B = std::make_unique<ExceptionBitVectorType>(N);
            H = sdsl::int_vector<>(h_count, 0, h);
            U last_high = 0;
            size_t h_idx = 0;
            uint64_t* b_data = B->raw_data_ptr();
            const U low_mask = (b > 0) ? ((U(1) << b) - 1) : 0;

            for (size_t i = 0; i < N; ++i) {
                const U element = static_cast<U>(S[i]) - static_cast<U>(base);
                const U high = element >> b;
                if (b > 0) L[i] = element & low_mask;
                if (i == 0 || high != last_high) {
                    H[h_idx++] = high;
                    // B was zero-initialized: unchanged high parts need no store.
                    constexpr bool reverse = ExceptionBitVectorType::reverse_bit_order;
                    const size_t bit = reverse ? 63 - (i & 63) : (i & 63);
                    b_data[i >> 6] |= uint64_t(1) << bit;
                }
                last_high = high;
            }
            assert(h_idx == h_count);
            if constexpr (RandomAccess) B->enable_rank();

        }

    private:
        template<uint8_t SPLIT_POINT>
        static uint64_t unpack_low(const uint64_t* words, size_t index) {
            if constexpr (SPLIT_POINT == 0) return 0;
            else if constexpr (SPLIT_POINT == 64) return words[index];
            else {
                const size_t position = index * SPLIT_POINT;
                const size_t offset = position & 63;
                uint64_t value = words[position >> 6] >> offset;
                if (offset + SPLIT_POINT > 64) {
                    value |= words[(position >> 6) + 1] << (64 - offset);
                }
                return value & ((uint64_t(1) << SPLIT_POINT) - 1);
            }
        }

        // Starting at an element index divisible by 64 makes all extraction
        // shifts and word indices compile-time constants for the next 64 values.
        template<uint8_t SPLIT_POINT, size_t INDEX>
        static uint64_t unpack_aligned_low(const uint64_t* words) {
            if constexpr (SPLIT_POINT == 0) return 0;
            else if constexpr (SPLIT_POINT == 64) return words[INDEX];
            else {
                constexpr size_t position = INDEX * SPLIT_POINT;
                constexpr size_t offset = position & 63;
                uint64_t value = words[position >> 6] >> offset;
                if constexpr (offset + SPLIT_POINT > 64) {
                    value |= words[(position >> 6) + 1] << (64 - offset);
                }
                return value & ((uint64_t(1) << SPLIT_POINT) - 1);
            }
        }

#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
        template<uint8_t SPLIT_POINT, size_t INDEX>
        static __m256i unpack_window4(const uint64_t* words) {
            static_assert(SPLIT_POINT > 0 && SPLIT_POINT <= 16 && INDEX + 4 <= 64);
            constexpr size_t word = (INDEX * SPLIT_POINT) / 64;
            constexpr unsigned bit = (INDEX * SPLIT_POINT) % 64;
            uint64_t window = words[word] >> bit;
            if constexpr (bit + 4 * SPLIT_POINT > 64)
                window |= words[word + 1] << (64 - bit);
            const __m256i shifts = _mm256_set_epi64x(3 * SPLIT_POINT, 2 * SPLIT_POINT, SPLIT_POINT, 0);
            const __m256i mask = _mm256_set1_epi64x((uint64_t(1) << SPLIT_POINT) - 1);
            return _mm256_and_si256(_mm256_srlv_epi64(
                _mm256_set1_epi64x(std::bit_cast<int64_t>(window)), shifts), mask);
        }
#endif

        template<uint8_t SPLIT_POINT, size_t... INDICES>
        static void unpack_aligned_block(const uint64_t* words, T* output,
                                         uint64_t offset, std::index_sequence<INDICES...>) {
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
            if constexpr (sizeof(T) == 8 && SPLIT_POINT > 0 && SPLIT_POINT <= 16 &&
                          sizeof...(INDICES) == 64) {
                const __m256i base_vector = _mm256_set1_epi64x(std::bit_cast<int64_t>(offset));
                ([&] {
                    if constexpr (INDICES % 4 == 0) {
                        const __m256i lows = unpack_window4<SPLIT_POINT, INDICES>(words);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + INDICES),
                            _mm256_add_epi64(base_vector, lows));
                    }
                }(), ...);
            } else
#endif
            {
                ((output[INDICES] = static_cast<T>(offset + unpack_aligned_low<SPLIT_POINT, INDICES>(words))), ...);
            }
        }

        template<uint8_t SPLIT_POINT>
        void unpack_low_range(size_t start, size_t end, T* output, uint64_t offset) const {
            if constexpr (SPLIT_POINT == 0) {
                std::fill_n(output, end - start, static_cast<T>(offset));
            } else {
                const uint64_t* words = L.data();
                size_t i = start;
                for (; i < end && (i & 63); ++i, ++output) {
                    *output = static_cast<T>(offset + unpack_low<SPLIT_POINT>(words, i));
                }
                for (; end - i >= 64; i += 64, output += 64) {
                    unpack_aligned_block<SPLIT_POINT>(words + ((i * SPLIT_POINT) >> 6),
                                                     output, offset, std::make_index_sequence<64>{});
                }
                for (; i < end; ++i, ++output) {
                    *output = static_cast<T>(offset + unpack_low<SPLIT_POINT>(words, i));
                }
            }
        }

        inline static constexpr auto byte_prefix_ranks = [] {
            std::array<std::array<uint8_t, 8>, 256> table{};
            for (size_t mask = 0; mask < table.size(); ++mask) {
                uint8_t rank = 0;
                for (size_t bit = 0; bit < 8; ++bit) {
                    rank += (mask >> bit) & 1;
                    table[mask][bit] = rank;
                }
            }
            return table;
        }();

#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
        template<uint8_t SPLIT_POINT, size_t... GROUPS>
        static void reconstruct_aligned_block(const uint64_t* words, uint64_t markers,
                                              const uint64_t* run_bases, T* output,
                                              std::index_sequence<GROUPS...>) {
            unsigned preceding = 0;
            ([&] {
                const auto& ranks = byte_prefix_ranks[(markers >> (GROUPS * 4)) & 15];
                const uint64_t* bases = run_bases + preceding;
                const __m256i highs = _mm256_set_epi64x(
                    std::bit_cast<int64_t>(bases[ranks[3]]), std::bit_cast<int64_t>(bases[ranks[2]]),
                    std::bit_cast<int64_t>(bases[ranks[1]]), std::bit_cast<int64_t>(bases[ranks[0]]));
                const __m256i lows = unpack_window4<SPLIT_POINT, GROUPS * 4>(words);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + GROUPS * 4),
                    _mm256_add_epi64(highs, lows));
                preceding += ranks[3];
            }(), ...);
        }
#endif

        template<uint8_t SPLIT_POINT>
        void reconstruct_marker_words(size_t count, T* output) const {
            using U = std::make_unsigned_t<T>;
            const uint64_t* bits = B->raw_data_ptr();
            size_t next_high = 0;
            uint64_t carry = static_cast<uint64_t>(base);
            auto next_base = [&] {
                uint64_t value = static_cast<uint64_t>(base);
                if constexpr (SPLIT_POINT < sizeof(T) * 8)
                    value += static_cast<uint64_t>(H[next_high++]) << SPLIT_POINT;
                return value;
            };
            std::array<uint64_t, 65> run_bases;
            size_t position = 0;
            for (; count - position >= 64; position += 64) {
                const uint64_t markers = bits[position >> 6];
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
                if constexpr (sizeof(T) == 8 && SPLIT_POINT > 0 && SPLIT_POINT <= 16) {
                    const uint64_t* words = L.data() + ((position * SPLIT_POINT) >> 6);
                    if (markers == 0) {
                        unpack_aligned_block<SPLIT_POINT>(words, output + position, carry,
                                                         std::make_index_sequence<64>{});
                        continue;
                    }
                    const unsigned runs = __builtin_popcountll(markers);
                    run_bases[0] = carry;
                    for (unsigned run = 1; run <= runs; ++run)
                        run_bases[run] = next_base();
                    reconstruct_aligned_block<SPLIT_POINT>(words, markers, run_bases.data(),
                                                          output + position, std::make_index_sequence<16>{});
                    carry = run_bases[runs];
                    continue;
                }
#endif
                if constexpr (SPLIT_POINT > 0) {
                    unpack_aligned_block<SPLIT_POINT>(L.data() + ((position * SPLIT_POINT) >> 6),
                        output + position, 0, std::make_index_sequence<64>{});
                }
                const unsigned runs = __builtin_popcountll(markers);
                run_bases[0] = carry;
                for (unsigned run = 1; run <= runs; ++run)
                    run_bases[run] = next_base();

                // Each byte maps its eight values to consecutive high parts.
                // The preceding rank carries runs across byte boundaries.
                unsigned preceding = 0;
                for (unsigned byte = 0; byte < 8; ++byte) {
                    const auto& ranks = byte_prefix_ranks[(markers >> (byte * 8)) & 255];
                    T* dst = output + position + byte * 8;
                    for (unsigned lane = 0; lane < 8; ++lane) {
                        const uint64_t value = run_bases[preceding + ranks[lane]];
                        if constexpr (SPLIT_POINT == 0)
                            dst[lane] = static_cast<T>(value);
                        else
                            dst[lane] = static_cast<T>(value + static_cast<U>(dst[lane]));
                    }
                    preceding += ranks[7];
                }
                carry = run_bases[runs];
            }
            if constexpr (SPLIT_POINT > 0)
                unpack_low_range<SPLIT_POINT>(position, count, output + position, 0);
            for (; position < count; ++position) {
                if ((bits[position >> 6] >> (position & 63)) & 1)
                    carry = next_base();
                if constexpr (SPLIT_POINT == 0)
                    output[position] = static_cast<T>(carry);
                else
                    output[position] = static_cast<T>(carry + static_cast<U>(output[position]));
            }
        }

        template<uint8_t SPLIT_POINT>
        [[gnu::flatten]] size_t get_elements_worker(size_t startIndex, size_t count, T* output) const {
            if (count == 0 || startIndex >= m_num_elements) [[unlikely]] return 0;
            const size_t actual = std::min(count, m_num_elements - startIndex);
            const size_t endIndex = startIndex + actual;

            if (h == 0) {
                unpack_low_range<SPLIT_POINT>(startIndex, endIndex, output, static_cast<uint64_t>(base));
                return actual;
            }

            if constexpr (!ExceptionBitVectorType::reverse_bit_order) {
                if (startIndex == 0 && actual == m_num_elements && actual >= 64) {
                    reconstruct_marker_words<SPLIT_POINT>(actual, output);
                    return actual;
                }
            }
            // Partial ranges retain aligned low-bit unpacking followed by
            // run reconstruction, without per-value bit-cursor branches.
            if constexpr (SPLIT_POINT > 0) {
                unpack_low_range<SPLIT_POINT>(startIndex, endIndex, output, 0);
            }
            size_t rank = 1;
            if (startIndex != 0) {
                if constexpr (RandomAccess) rank = B->rank_unchecked(startIndex + 1);
            }
            const uint64_t* bits = B->raw_data_ptr();
            const size_t word_limit = (endIndex + 63) / 64;
            constexpr bool reverse = ExceptionBitVectorType::reverse_bit_order;
            const size_t search = startIndex + 1;
            size_t word_index = search >> 6;
            const uint64_t mask = reverse ? (~uint64_t(0) >> (search & 63))
                                          : (~uint64_t(0) << (search & 63));
            uint64_t word = word_index < word_limit ? bits[word_index] & mask : 0;
            size_t position = startIndex;
            while (position < endIndex) {
                while (word == 0 && word_index + 1 < word_limit)
                    word = bits[++word_index];
                size_t next_run = endIndex;
                if (word != 0) {
                    const size_t bit = reverse ? __builtin_clzll(word) : __builtin_ctzll(word);
                    next_run = std::min(endIndex, word_index * 64 + bit);
                    if constexpr (reverse) word &= ~(uint64_t(1) << (63 - bit));
                    else word &= word - 1;
                }
                uint64_t run_base = static_cast<uint64_t>(base);
                if constexpr (SPLIT_POINT < sizeof(T) * 8) {
                    run_base += static_cast<uint64_t>(H[rank - 1]) << SPLIT_POINT;
                }
                T* first = output + (position - startIndex);
                T* last = output + (next_run - startIndex);
                if constexpr (SPLIT_POINT == 0) {
                    std::fill(first, last, static_cast<T>(run_base));
                } else {
                    using U = std::make_unsigned_t<T>;
                    for (; first != last; ++first) {
                        *first = static_cast<T>(run_base + static_cast<U>(*first));
                    }
                }
                position = next_run;
                ++rank;
            }
            return actual;
        }

        template <size_t... Is>
        size_t dispatch_worker(size_t b_val, size_t start, size_t count, T* out, std::index_sequence<Is...>) const {
             using WorkerPtr = size_t (RLE_GEF::*)(size_t, size_t, T*) const;
             static constexpr WorkerPtr table[] = { &RLE_GEF::get_elements_worker<Is>... };
             if (b_val >= sizeof...(Is)) throw std::invalid_argument("Invalid b value");
             return (this->*table[b_val])(start, count, out);
        }

    public:
        // The caller provides room for at least min(count, size()-startIndex)
        // values. No writes occur for an empty request or startIndex >= size().
        size_t get_elements_into(size_t startIndex, size_t count, T* output) const {
            if (count == 0 || startIndex >= m_num_elements) return 0;
            const size_t actual = std::min(count, m_num_elements - startIndex);
            if constexpr (RandomAccess) {
                return dispatch_worker(b, startIndex, actual, output, std::make_index_sequence<sizeof(T) * 8 + 1>{});
            } else {
                // A support-free range must still decode the preceding prefix.
                if (startIndex == 0) {
                    return dispatch_worker(b, 0, actual, output, std::make_index_sequence<sizeof(T) * 8 + 1>{});
                }
                std::vector<T> prefix(startIndex + actual);
                dispatch_worker(b, 0, prefix.size(), prefix.data(), std::make_index_sequence<sizeof(T) * 8 + 1>{});
                std::copy_n(prefix.data() + startIndex, actual, output);
                return actual;
            }
        }

        size_t get_elements(size_t startIndex, size_t count, std::vector<T>& output) const override {
            if (count == 0 || startIndex >= m_num_elements) return 0;
            if (output.size() < count) {
                throw std::invalid_argument("output buffer is smaller than requested count");
            }
            return get_elements_into(startIndex, count, output.data());
        }

        [[gnu::flatten]] inline T operator[](size_t index) const override {
            if constexpr (!RandomAccess) {
                std::vector<T> scratch;
                return ::gef::internal::no_support::at_force_from_zero<T>(
                    index, scratch,
                    [this](size_t n, std::vector<T>& out) { return this->get_elements_into(0, n, out.data()); });
            }
            // Case 1: No high bits are used (h=0).
            // All information is stored in the L vector. Reconstruction is trivial.
            if (h == 0) [[unlikely]] {
                if (b == 0) [[unlikely]] return base;
                return base + L[index];
            }

            using U = std::make_unsigned_t<T>;
            // Initiate L access early to hide memory latency
            const U low = (b > 0) ? static_cast<U>(L[index]) : U(0);

            // Rank identifies the containing run for both a run's first value
            // and its remaining values; inspecting B[index] is redundant.
            const size_t rank = B->rank_unchecked(index + 1);
            const U high = static_cast<U>(H[rank - 1]);
            return static_cast<T>(static_cast<U>(base) + (low | (high << b)));
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
            if (h > 0)
                B->serialize(ofs);
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
            
            H.load(ifs);
            if (h > 0) {
                B = std::make_unique<ExceptionBitVectorType>(ExceptionBitVectorType::load(ifs));
                if constexpr (RandomAccess) {
                    B->enable_rank();
                }

            } else {
                B = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return m_num_elements;
        }

        [[nodiscard]] size_t size_in_bytes_without_supports() const override {
            size_t total_bytes = 0;
            if (B) {
                total_bytes += B->size_in_bytes() - B->support_size_in_bytes();
            }
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sdsl::size_in_bytes(H);
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
            
            // B bit vector (if it exists)
            if (B) {
                total_bytes += bits_to_bytes(B->size());
            }
            
            // Fixed metadata
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            
            return total_bytes;
        }

        // Owned C++ object storage and dynamic capacities; allocator metadata
        // and allocation-size-class rounding are intentionally excluded.
        [[nodiscard]] size_t allocated_size_in_bytes() const
            requires requires(const ExceptionBitVectorType& bits) { bits.allocated_size_in_bytes(); } {
            return sizeof(*this) + sdsl_allocated_payload_bytes(H) +
                sdsl_allocated_payload_bytes(L) + (B ? B->allocated_size_in_bytes() : 0);
        }

        [[nodiscard]] size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (B) {
                total_bytes += B->size_in_bytes();
            }
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sdsl::size_in_bytes(H);
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            return total_bytes;
        }

        [[nodiscard]] uint8_t split_point() const override {
            return this -> b;
        }


    };
    } // namespace internal
} // namespace gef

#endif


