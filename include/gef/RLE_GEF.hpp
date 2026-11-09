#ifndef RLE_GEF_H
#define RLE_GEF_H

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
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
#include "../datastructures/HalfBlockRank.hpp"

#if (defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)) || defined(__BMI2__)
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
    class alignas(64) RLE_GEF : public IGEF<T> {
        // Two zero words standing in for L when b == 0, so a low part can be
        // read without a branch on b.
        static constexpr uint64_t NO_WORDS[2] = {0, 0};

    public:
        // Everything operator[] reads comes first, so that with the cache-line
        // alignment of the class a query touches one line of the object.

        // Rank over B for random access when B is small enough; larger B
        // use the bitvector's own rank support.
        HalfBlockRank run_rank;

        // L.data() (or NO_WORDS when b == 0) and H.data(); see refresh_views().
        const uint64_t* low_words = NO_WORDS;
        const uint64_t* high_words = NO_WORDS;

        /**
         * The minimum of the encoded sequence, so that we store the shifted sequence
         * that falls in the range [0, max S - base]
         * This tricks may boost compression and allows us to implicitly store negative numbers
         */
        T base;

        // The split point that rules which bits are stored in H and in L
        uint8_t b;
        uint8_t h;

        // Bit-vector such that B[i] = 1 <==> highPart(i) != highPart(i - 1)
        std::unique_ptr<ExceptionBitVectorType> B;

        // high parts
        sdsl::int_vector<> H;

        // low parts
        sdsl::int_vector<> L;

        size_t m_num_elements;

        // Re-derives the cached word pointers after L or H change.
        void refresh_views() {
            low_words = b > 0 ? L.data() : NO_WORDS;
            high_words = H.data();
        }

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

        static constexpr bool own_rank_capable =
            RandomAccess && !ExceptionBitVectorType::reverse_bit_order &&
            requires(const ExceptionBitVectorType& bits) { { bits.raw_data_ptr() } -> std::convertible_to<const uint64_t*>; };

        void enable_random_access() {
            if constexpr (RandomAccess) {
                run_rank.clear();
                if constexpr (own_rank_capable) {
                    if (HalfBlockRank::supports(B->size())) {
                        run_rank.build(B->raw_data_ptr(), B->size());
                        // Streams written by earlier versions carry the
                        // bitvector's own support; it would be unused here.
                        if constexpr (requires { B->disable_support(); }) B->disable_support();
                        return;
                    }
                }
                B->enable_rank();
            }
        }

        // Number of runs starting in [0, index): ones of B before index.
        [[gnu::always_inline]] size_t runs_before(size_t index) const {
            if (run_rank.enabled()) [[likely]] return run_rank.rank1(index);
            return B->rank_unchecked(index);
        }

    public:
        using IGEF<T>::serialize;
        using IGEF<T>::load;

        ~RLE_GEF() override = default;

        // Default constructor
        RLE_GEF() : base(0), b(0), h(0), m_num_elements(0) {
        }

        // 2. Copy Constructor
        RLE_GEF(const RLE_GEF &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              base(other.base),
              b(other.b),
              h(other.h),
              H(other.H),
              L(other.L),
              m_num_elements(other.m_num_elements) {
            if (other.h > 0) {
                B = std::make_unique<ExceptionBitVectorType>(*other.B);
                enable_random_access();

            } else {
                B = nullptr;
            }
            refresh_views();
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
            swap(first.run_rank, second.run_rank);
            swap(first.low_words, second.low_words);
            swap(first.high_words, second.high_words);
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
              run_rank(std::move(other.run_rank)),
              base(other.base),
              b(other.b),
              h(other.h),
              B(std::move(other.B)),
              H(std::move(other.H)),
              L(std::move(other.L)),
              m_num_elements(other.m_num_elements) {
            // Leave the moved-from object in a valid, empty state
            other.h = 0;
            other.m_num_elements = 0;
            other.base = T{};
            // The vectors moved with their buffers; re-derive both views.
            refresh_views();
            other.refresh_views();
        }


        // 5. Move Assignment Operator
        RLE_GEF &operator=(RLE_GEF &&other) noexcept {
            if (this != &other) {
                B = std::move(other.B);
                H = std::move(other.H);
                L = std::move(other.L);
                run_rank = std::move(other.run_rank);
                h = other.h;
                b = other.b;
                m_num_elements = other.m_num_elements;
                base = other.base;
                refresh_views();
                other.refresh_views();
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
                refresh_views();
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
                // Independent reductions avoid the pairwise comparison branch
                // in minmax_element and allow contiguous inputs to vectorize.
                auto minimum = *S.begin();
                auto maximum = minimum;
                for (const auto value : S) {
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
                base = minimum;
                // Integer width avoids rounding at powers of two and overflow of
                // max-min+1. A constant block needs only its stored base.
                const U range = static_cast<U>(maximum) - static_cast<U>(base);
                total_bits = static_cast<uint8_t>(std::bit_width(range));
                b = optimal_split_point(S, total_bits, base, &h_count);
            }
            h = total_bits - b;

            if (b > 0) {
                L = sdsl::int_vector<>(S.size(), 0, b);
            } else {
                L = sdsl::int_vector<>(0);
            }

            // Both packed streams are filled sequentially. Buffer each word
            // so a completed word needs only one store instead of proxy updates.
            struct PackedWriter {
                uint64_t* data;
                unsigned width;
                uint64_t buffer = 0;
                unsigned bits = 0;

                void append(uint64_t value) {
                    buffer |= value << bits;
                    const unsigned filled = bits + width;
                    if (filled >= 64) {
                        *data++ = buffer;
                        buffer = filled > 64 ? value >> (64 - bits) : 0;
                        bits = filled - 64;
                    } else {
                        bits = filled;
                    }
                }

                void finish() { if (bits) *data = buffer; }
            };
            PackedWriter low_writer{L.data(), b};

            if (h == 0) {
                // All in L - use unsigned arithmetic for efficiency
                using U = std::make_unsigned_t<T>;
                
                if (b > 0) {
                    for (size_t i = 0; i < S.size(); i++) {
                        low_writer.append(static_cast<U>(static_cast<U>(S[i]) - static_cast<U>(base)));
                    }
                }
                low_writer.finish();
                B = nullptr;
                H.resize(0);
                refresh_views();
                return;
            }

            // The split histogram already gives the number of runs. Allocate
            // H exactly once and fill H, L and B in the same source scan.
            B = std::make_unique<ExceptionBitVectorType>(N);
            H = sdsl::int_vector<>(h_count, 0, h);
            PackedWriter high_writer{H.data(), h};
            U last_high = 0;
            size_t h_idx = 0;
            uint64_t* b_data = B->raw_data_ptr();
            const U low_mask = (b > 0) ? ((U(1) << b) - 1) : 0;

            for (size_t i = 0; i < N; ++i) {
                const U element = static_cast<U>(S[i]) - static_cast<U>(base);
                const U high = element >> b;
                if (b > 0) low_writer.append(element & low_mask);
                if (i == 0 || high != last_high) {
                    high_writer.append(high);
                    ++h_idx;
                    // B was zero-initialized: unchanged high parts need no store.
                    constexpr bool reverse = ExceptionBitVectorType::reverse_bit_order;
                    const size_t bit = reverse ? 63 - (i & 63) : (i & 63);
                    b_data[i >> 6] |= uint64_t(1) << bit;
                }
                last_high = high;
            }
            low_writer.finish();
            high_writer.finish();
            assert(h_idx == h_count);
            enable_random_access();
            refresh_views();
        }

    private:
        // Field `index` of width-bit values packed in `words` (width = 0 reads
        // 0 from two readable words) without data-dependent branches: the
        // field may span two words, and sdsl allocates one word past the last.
        [[gnu::always_inline]] static uint64_t read_field(const uint64_t* words, uint8_t width, size_t index) {
            const uint64_t bit = static_cast<uint64_t>(index) * width;
#if defined(__BMI2__)
            if constexpr (std::endian::native == std::endian::little) {
                // Fields of at most 57 bits lie within the eight bytes starting
                // at their first byte: one unaligned load. Widths are fixed per
                // vector, so this branch is predicted.
                if (width <= 57) [[likely]] {
                    uint64_t bytes;
                    std::memcpy(&bytes, reinterpret_cast<const char*>(words) + (bit >> 3), sizeof(bytes));
                    return _bzhi_u64(bytes >> (bit & 7), width);
                }
            }
#endif
            const unsigned offset = bit & 63;
            const uint64_t* word = words + (bit >> 6);
            const uint64_t window = (word[0] >> offset) | ((word[1] << 1) << (63 - offset));
#if defined(__BMI2__)
            return _bzhi_u64(window, width); // width >= 64 keeps every bit
#else
            return width >= 64 ? window : window & ((uint64_t(1) << width) - 1);
#endif
        }

        template<uint8_t SPLIT_POINT>
        static uint64_t unpack_low(const uint64_t* words, size_t index) {
            if constexpr (SPLIT_POINT == 0) return 0;
            else if constexpr (SPLIT_POINT == 64) return words[index];
            else {
                const uint64_t position = uint64_t(index) * SPLIT_POINT;
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
        template<uint8_t SPLIT_POINT, unsigned PHASE, size_t BYTE_BIAS>
        static __m256i wide_low_shuffle() {
            static_assert(PHASE + 4 * SPLIT_POINT + 8 * BYTE_BIAS <= 128);
            // Interior groups share one mask (two for odd widths); only the
            // final group can need a left-shifted source window.
            static constexpr auto shuffle = [] {
                std::array<uint8_t, 32> result{};
                result.fill(0x80);
                for (unsigned lane = 0; lane < 4; ++lane) {
                    const unsigned bit = PHASE + lane * SPLIT_POINT;
                    const unsigned bytes = ((bit & 7) + SPLIT_POINT + 7) / 8;
                    for (unsigned byte = 0; byte < bytes; ++byte)
                        result[lane * 8 + byte] = BYTE_BIAS + bit / 8 + byte;
                }
                return result;
            }();
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(shuffle.data()));
        }
#endif

        template<uint8_t SPLIT_POINT, size_t... INDICES>
        static void unpack_aligned_block(const uint64_t* words, T* output,
                                         uint64_t offset, std::index_sequence<INDICES...>) {
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
            if constexpr (sizeof(T) == 8 && sizeof...(INDICES) == 64 &&
                          (SPLIT_POINT == 8 || SPLIT_POINT == 16 || SPLIT_POINT == 32)) {
                const auto* bytes = reinterpret_cast<const unsigned char*>(words);
                const __m256i base_vector = _mm256_set1_epi64x(std::bit_cast<int64_t>(offset));
                ([&] {
                    if constexpr (INDICES % 4 == 0) {
                        __m256i lows;
                        if constexpr (SPLIT_POINT == 8) {
                            lows = _mm256_cvtepu8_epi64(_mm_loadu_si32(bytes + INDICES));
                        } else if constexpr (SPLIT_POINT == 16) {
                            lows = _mm256_cvtepu16_epi64(_mm_loadl_epi64(
                                reinterpret_cast<const __m128i*>(bytes + INDICES * 2)));
                        } else {
                            lows = _mm256_cvtepu32_epi64(_mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(bytes + INDICES * 4)));
                        }
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + INDICES),
                            _mm256_add_epi64(base_vector, lows));
                    }
                }(), ...);
            } else if constexpr (sizeof(T) == 8 && SPLIT_POINT > 16 && SPLIT_POINT < 32 &&
                                 sizeof...(INDICES) == 64) {
                const auto* bytes = reinterpret_cast<const unsigned char*>(words);
                const __m256i base_vector = _mm256_set1_epi64x(std::bit_cast<int64_t>(offset));
                const __m256i mask = _mm256_set1_epi64x((uint64_t(1) << SPLIT_POINT) - 1);
                ([&] {
                    if constexpr (INDICES % 4 == 0) {
                        constexpr size_t first_byte = (INDICES * SPLIT_POINT) / 8;
                        constexpr unsigned phase = (INDICES * SPLIT_POINT) & 7;
                        // A complete low block occupies exactly 8*b bytes.
                        // Clamp the final 16-byte load inside that block.
                        constexpr size_t load_byte = std::min(first_byte, size_t(8 * SPLIT_POINT - 16));
                        const __m256i source = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(bytes + load_byte)));
                        const __m256i shifts = _mm256_set_epi64x((phase + 3 * SPLIT_POINT) & 7,
                            (phase + 2 * SPLIT_POINT) & 7, (phase + SPLIT_POINT) & 7, phase);
                        const __m256i lows = _mm256_and_si256(_mm256_srlv_epi64(
                            _mm256_shuffle_epi8(source,
                                wide_low_shuffle<SPLIT_POINT, phase, first_byte - load_byte>()),
                            shifts), mask);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + INDICES),
                            _mm256_add_epi64(base_vector, lows));
                    }
                }(), ...);
            } else if constexpr (sizeof(T) == 8 && SPLIT_POINT > 0 && SPLIT_POINT <= 16 &&
                          sizeof...(INDICES) == 64) {
                const __m256i base_vector = _mm256_set1_epi64x(std::bit_cast<int64_t>(offset));
                const __m256i shifts = _mm256_set_epi64x(3 * SPLIT_POINT, 2 * SPLIT_POINT, SPLIT_POINT, 0);
                const __m256i mask = _mm256_set1_epi64x((uint64_t(1) << SPLIT_POINT) - 1);
                ([&] {
                    if constexpr (INDICES % 4 == 0) {
                        // Four lows fit in one window. A crossing window's
                        // second word remains inside this complete 64-value block.
                        constexpr size_t word = (INDICES * SPLIT_POINT) / 64;
                        constexpr unsigned bit = (INDICES * SPLIT_POINT) % 64;
                        uint64_t window = words[word] >> bit;
                        if constexpr (bit + 4 * SPLIT_POINT > 64)
                            window |= words[word + 1] << (64 - bit);
                        const __m256i lows = _mm256_and_si256(_mm256_srlv_epi64(
                            _mm256_set1_epi64x(std::bit_cast<int64_t>(window)), shifts), mask);
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
                    unpack_aligned_block<SPLIT_POINT>(words + ((uint64_t(i) * SPLIT_POINT) >> 6),
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
        template<uint8_t SPLIT_POINT, size_t INDEX>
        static void reconstruct_window4(const uint64_t* words, __m256i highs, T* output) {
            constexpr size_t word = (INDEX * SPLIT_POINT) / 64;
            constexpr unsigned bit = (INDEX * SPLIT_POINT) % 64;
            uint64_t window = words[word] >> bit;
            if constexpr (bit + 4 * SPLIT_POINT > 64)
                window |= words[word + 1] << (64 - bit);
            const __m256i shifts = _mm256_set_epi64x(3 * SPLIT_POINT, 2 * SPLIT_POINT,
                                                    SPLIT_POINT, 0);
            const __m256i mask = _mm256_set1_epi64x((uint64_t(1) << SPLIT_POINT) - 1);
            const __m256i lows = _mm256_and_si256(_mm256_srlv_epi64(
                _mm256_set1_epi64x(std::bit_cast<int64_t>(window)), shifts), mask);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + INDEX),
                               _mm256_add_epi64(highs, lows));
        }

        template<uint8_t SPLIT_POINT, size_t INDEX>
        static void reconstruct_window4(const uint64_t* words, const uint64_t* bases,
                                        const uint8_t* ranks, T* output) {
            // Every selected rank refers to an initialized high base. Unlike
            // a contiguous vector load, this needs no padding after the last run.
            const __m256i highs = _mm256_set_epi64x(
                std::bit_cast<int64_t>(bases[ranks[3]]),
                std::bit_cast<int64_t>(bases[ranks[2]]),
                std::bit_cast<int64_t>(bases[ranks[1]]),
                std::bit_cast<int64_t>(bases[ranks[0]]));
            reconstruct_window4<SPLIT_POINT, INDEX>(words, highs, output);
        }

#if defined(__AVX512F__)
        template<uint8_t SPLIT_POINT, size_t INDEX>
        static void reconstruct_window8(const uint64_t* words, __m512i highs, T* output) {
            static_assert(sizeof(T) == 8 && SPLIT_POINT >= 1 && SPLIT_POINT <= 16 && INDEX % 8 == 0 && INDEX <= 56);
            constexpr size_t word = (INDEX * SPLIT_POINT) / 64;
            constexpr unsigned bit = (INDEX * SPLIT_POINT) % 64;
            constexpr unsigned window_values = SPLIT_POINT <= 8 ? 8 : 4;
            uint64_t window = words[word] >> bit;
            if constexpr (bit + window_values * SPLIT_POINT > 64)
                window |= words[word + 1] << (64 - bit);
            __m512i source = _mm512_set1_epi64(std::bit_cast<int64_t>(window));
            if constexpr (SPLIT_POINT > 8) {
                // Each half retains the bounded four-value packed window.
                constexpr size_t upper_word = ((INDEX + 4) * SPLIT_POINT) / 64;
                constexpr unsigned upper_bit = ((INDEX + 4) * SPLIT_POINT) % 64;
                uint64_t upper = words[upper_word] >> upper_bit;
                if constexpr (upper_bit + 4 * SPLIT_POINT > 64)
                    upper |= words[upper_word + 1] << (64 - upper_bit);
                source = _mm512_mask_blend_epi64(0xf0, source,
                    _mm512_set1_epi64(std::bit_cast<int64_t>(upper)));
            }
            constexpr unsigned upper_shift = SPLIT_POINT <= 8 ? 4 * SPLIT_POINT : 0;
            const __m512i shifts = _mm512_set_epi64(upper_shift + 3 * SPLIT_POINT,
                upper_shift + 2 * SPLIT_POINT, upper_shift + SPLIT_POINT, upper_shift,
                3 * SPLIT_POINT, 2 * SPLIT_POINT, SPLIT_POINT, 0);
            const __m512i mask = _mm512_set1_epi64((uint64_t(1) << SPLIT_POINT) - 1);
            const __m512i lows = _mm512_and_si512(_mm512_srlv_epi64(
                source, shifts), mask);
            _mm512_storeu_si512(output + INDEX, _mm512_add_epi64(highs, lows));
        }

        static __m512i map_high_byte(const uint64_t* bases, const uint8_t* ranks, unsigned markers) {
            // Rank8 needs the ninth base; dense bytes are already in output order.
            if (markers == 255) return _mm512_loadu_si512(bases + 1);
            const __m512i indices = _mm512_cvtepu8_epi64(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ranks)));
            return _mm512_permutexvar_epi64(indices, _mm512_loadu_si512(bases));
        }
#endif

        template<uint8_t SPLIT_POINT, size_t... BYTES>
        static void reconstruct_fused_block(const uint64_t* words, uint64_t markers,
                                            const uint64_t* bases, T* output,
                                            std::index_sequence<BYTES...>) {
            unsigned preceding = 0;
            ([&] {
                const unsigned byte = (markers >> (BYTES * 8)) & 255;
                const auto& ranks = byte_prefix_ranks[byte];
#if defined(__AVX512F__)
                if constexpr (sizeof(T) == 8) {
                    const __m512i highs = map_high_byte(bases + preceding, ranks.data(), byte);
                    reconstruct_window8<SPLIT_POINT, BYTES * 8>(words, highs, output);
                } else
#endif
                {
                    reconstruct_window4<SPLIT_POINT, BYTES * 8>(
                        words, bases + preceding, ranks.data(), output);
                    reconstruct_window4<SPLIT_POINT, BYTES * 8 + 4>(
                        words, bases + preceding, ranks.data() + 4, output);
                }
                preceding += ranks[7];
            }(), ...);
        }
#endif

        template<uint8_t SPLIT_POINT, bool Partial = false>
        void reconstruct_marker_words(size_t count, T* output, size_t startIndex = 0) const {
            using U = std::make_unsigned_t<T>;
            const uint64_t* bits = B->raw_data_ptr();
            const uint64_t* high_word = H.data();
            const unsigned high_width = H.width();
            const uint64_t high_mask = high_width == 64 ? ~uint64_t(0)
                : (uint64_t(1) << high_width) - 1;
            unsigned high_offset = 0;
            uint64_t carry = static_cast<uint64_t>(base);
            const size_t begin = Partial ? startIndex : 0;
            if constexpr (Partial) {
                count += begin;
                if constexpr (RandomAccess) {
                    if (begin != 0) {
                        // Cursor precedes B[begin], which the decoder consumes
                        // inclusively even when begin is a marker-word boundary.
                        const size_t preceding = runs_before(begin);
                        const uint64_t high_bit = uint64_t(preceding) * high_width;
                        high_word += high_bit >> 6;
                        high_offset = high_bit & 63;
                        if (preceding != 0)
                            carry += static_cast<uint64_t>(H[preceding - 1]) << SPLIT_POINT;
                    }
                }
            }
            auto next_base = [&] {
                uint64_t value = static_cast<uint64_t>(base);
                if constexpr (SPLIT_POINT < sizeof(T) * 8) {
                    uint64_t high;
                    if (high_width == 64) {
                        high = *high_word++;
                    } else {
                        high = *high_word >> high_offset;
                        const unsigned end = high_offset + high_width;
                        // Read the next word only when this valid field crosses.
                        if (end > 64)
                            high |= high_word[1] << (64 - high_offset);
                        high_word += end >> 6;
                        high_offset = end & 63;
                        high &= high_mask;
                    }
                    value += high << SPLIT_POINT;
                }
                return value;
            };
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
            const __m256i high_shifts = _mm256_set_epi64x(3 * high_width, 2 * high_width,
                                                         high_width, 0);
            const __m256i high_mask_vector = _mm256_set1_epi64x(std::bit_cast<int64_t>(high_mask));
            const __m256i base_vector = _mm256_set1_epi64x(
                std::bit_cast<int64_t>(static_cast<uint64_t>(base)));
            const unsigned group_bits = 4 * high_width;
#endif
            auto fill_high_bases = [&](uint64_t* destination, unsigned count) {
                unsigned filled = 0;
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
                if constexpr (sizeof(T) == 8 && SPLIT_POINT < 64) {
                    if (high_width <= 16) {
                        for (; count - filled >= 4; filled += 4) {
                            // Four real fields fit in one 64-bit window. Only
                            // their actual crossing can require a second word.
                            uint64_t window = *high_word >> high_offset;
                            const unsigned end = high_offset + group_bits;
                            if (end > 64)
                                window |= high_word[1] << (64 - high_offset);
                            high_word += end >> 6;
                            high_offset = end & 63;
                            const __m256i highs = _mm256_and_si256(_mm256_srlv_epi64(
                                _mm256_set1_epi64x(std::bit_cast<int64_t>(window)), high_shifts),
                                high_mask_vector);
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + filled),
                                _mm256_add_epi64(base_vector, _mm256_slli_epi64(highs, SPLIT_POINT)));
                        }
                    }
                }
#endif
                for (; filled < count; ++filled) destination[filled] = next_base();
            };
            std::array<uint64_t, 65> run_bases;
#if defined(__AVX2__) && defined(__AVX512F__) && !defined(GEF_DISABLE_SIMD)
            // Wide loads may include unselected trailing slots, never selected padding.
            if constexpr (sizeof(T) == 8) run_bases.fill(0);
#endif
            size_t position = begin;
            if constexpr (Partial) {
                for (; position < count && (position & 63); ++position) {
                    if ((bits[position >> 6] >> (position & 63)) & 1)
                        carry = next_base();
                    output[position - begin] = static_cast<T>(carry +
                        unpack_low<SPLIT_POINT>(L.data(), position));
                }
            }
            for (; count - position >= 64; position += 64) {
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
                if constexpr (sizeof(T) == 8 && SPLIT_POINT > 0 && SPLIT_POINT <= 16) {
                    const uint64_t markers = bits[position >> 6];
                    const unsigned runs = __builtin_popcountll(markers);
                    run_bases[0] = carry;
                    fill_high_bases(run_bases.data() + 1, runs);
                    // Unpack and reconstruct before the only output store.
                    reconstruct_fused_block<SPLIT_POINT>(
                        L.data() + ((uint64_t(position) * SPLIT_POINT) >> 6), markers,
                        run_bases.data(), output + (position - begin), std::make_index_sequence<8>{});
                    carry = run_bases[runs];
                    continue;
                }
#endif
                // Reconstruct while this aligned low-bit block is still hot.
                if constexpr (SPLIT_POINT > 0) {
                    unpack_aligned_block<SPLIT_POINT>(L.data() + ((uint64_t(position) * SPLIT_POINT) >> 6),
                        output + (position - begin), 0, std::make_index_sequence<64>{});
                }
                const uint64_t markers = bits[position >> 6];
                const unsigned runs = __builtin_popcountll(markers);
                run_bases[0] = carry;
                fill_high_bases(run_bases.data() + 1, runs);

                // Each byte maps its eight values to consecutive high parts.
                // The preceding rank carries runs across byte boundaries.
                unsigned preceding = 0;
                for (unsigned byte = 0; byte < 8; ++byte) {
                    const unsigned marker_byte = (markers >> (byte * 8)) & 255;
                    const auto& ranks = byte_prefix_ranks[marker_byte];
                    T* dst = output + (position - begin) + byte * 8;
#if defined(__AVX2__) && defined(__AVX512F__) && !defined(GEF_DISABLE_SIMD)
                    if constexpr (sizeof(T) == 8) {
                        __m512i values = map_high_byte(run_bases.data() + preceding, ranks.data(), marker_byte);
                        if constexpr (SPLIT_POINT > 0)
                            values = _mm512_add_epi64(values, _mm512_loadu_si512(dst));
                        _mm512_storeu_si512(dst, values);
                    } else
#endif
                    {
                        for (unsigned lane = 0; lane < 8; ++lane) {
                            const uint64_t value = run_bases[preceding + ranks[lane]];
                            if constexpr (SPLIT_POINT == 0)
                                dst[lane] = static_cast<T>(value);
                            else
                                dst[lane] = static_cast<T>(value + static_cast<U>(dst[lane]));
                        }
                    }
                    preceding += ranks[7];
                }
                carry = run_bases[runs];
            }
            if constexpr (SPLIT_POINT > 0 && !Partial)
                unpack_low_range<SPLIT_POINT>(position, count, output + position, 0);
            for (; position < count; ++position) {
                if ((bits[position >> 6] >> (position & 63)) & 1)
                    carry = next_base();
                if constexpr (Partial)
                    output[position - begin] = static_cast<T>(carry +
                        unpack_low<SPLIT_POINT>(L.data(), position));
                else if constexpr (SPLIT_POINT == 0)
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
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
                if constexpr (sizeof(T) == 8 && SPLIT_POINT <= 16) {
                    // Use the marker kernel when a complete aligned word
                    // remains after the scalar prefix.
                    if (actual >= 64 && ((64 - (startIndex & 63)) & 63) <= actual - 64) {
                        reconstruct_marker_words<SPLIT_POINT, true>(actual, output, startIndex);
                        return actual;
                    }
                }
#endif
            }
            // Partial ranges retain aligned low-bit unpacking followed by
            // run reconstruction, without per-value bit-cursor branches.
            if constexpr (SPLIT_POINT > 0) {
                unpack_low_range<SPLIT_POINT>(startIndex, endIndex, output, 0);
            }
            size_t rank = 1;
            if (startIndex != 0) {
                if constexpr (RandomAccess) rank = runs_before(startIndex + 1);
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
            using U = std::make_unsigned_t<T>;
            const U low = static_cast<U>(read_field(low_words, b, index));
            // No high bits (h == 0): the low part is the whole offset.
            if (h == 0) [[unlikely]]
                return static_cast<T>(static_cast<U>(base) + low);

            // Rank identifies the containing run for both a run's first value
            // and its remaining values; inspecting B[index] is redundant.
            const size_t rank = runs_before(index + 1);
            const U high = static_cast<U>(read_field(high_words, h, rank - 1));
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
                enable_random_access();
            } else {
                B = nullptr;
                run_rank.clear();
            }
            refresh_views();
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
                sdsl_allocated_payload_bytes(L) + (B ? B->allocated_size_in_bytes() : 0) +
                run_rank.allocated_size_in_bytes();
        }

        [[nodiscard]] size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (B) {
                total_bytes += B->size_in_bytes();
                total_bytes += run_rank.size_in_bytes();
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


