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
#if (defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)) || defined(__BMI2__)
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
#include "../datastructures/PairedZeroSelect.hpp"
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

        // Random-access select0 index over G_plus/G_minus for partitions with
        // few enough elements; larger ones use the bitvectors' own select0.
        PairedZeroSelect zero_select;

        static constexpr bool compact_select_capable =
            RandomAccess && !GapBitVectorType::reverse_bit_order &&
            requires(const GapBitVectorType& bits) { { bits.raw_data_ptr() } -> std::convertible_to<const uint64_t*>; };

        void enable_random_access() {
            if constexpr (RandomAccess) {
                zero_select.clear();
                if constexpr (compact_select_capable) {
                    if (PairedZeroSelect::supports(m_num_elements, G_plus->size(), G_minus->size())) {
                        zero_select.build(G_plus->raw_data_ptr(), G_plus->size(),
                                          G_minus->raw_data_ptr(), G_minus->size(), m_num_elements);
                        // Streams written before this index existed carry the
                        // bitvectors' own support; it would be unused here.
                        if constexpr (requires { G_plus->disable_support(); }) {
                            G_plus->disable_support();
                            G_minus->disable_support();
                        }
                        return;
                    }
                }
                G_plus->enable_select0();
                G_minus->enable_select0();
            }
        }

        // L[index] without data-dependent branches: the field may span two
        // words, and sdsl allocates one word past the last one. With b == 0 a
        // zero mask discards a read from a static two-word stand-in.
        [[gnu::always_inline]] uint64_t read_low_unchecked(size_t index) const {
            static constexpr uint64_t no_low_words[2] = {0, 0};
            // Select the pointer arithmetically: compilers otherwise branch on
            // b, which mispredicts when partitions with b == 0 are interleaved.
            const uintptr_t has_low = uintptr_t(0) - static_cast<uintptr_t>(b != 0);
            const uint64_t* words = reinterpret_cast<const uint64_t*>(
                (reinterpret_cast<uintptr_t>(L.data()) & has_low) |
                (reinterpret_cast<uintptr_t>(no_low_words) & ~has_low));
            const uint64_t bit = static_cast<uint64_t>(index) * b;
            const unsigned offset = bit & 63;
            const uint64_t* word = words + (bit >> 6);
            const uint64_t window = (word[0] >> offset) | ((word[1] << 1) << (63 - offset));
#if defined(__BMI2__)
            return _bzhi_u64(window, b); // b >= 64 keeps every bit
#else
            return b >= 64 ? window : window & ((uint64_t(1) << b) - 1);
#endif
        }

        // Position of the rank-th (1-based) zero of G_plus (side 0) or G_minus (side 1).
        size_t gap_select0(size_t side, size_t rank) const {
            if (zero_select.enabled()) return zero_select.select0(side, rank);
            return side ? G_minus->select0(rank) : G_plus->select0(rank);
        }

        [[gnu::noinline]] std::pair<size_t, size_t> gap_select0_pair_general(size_t rank) const {
            if constexpr (requires { G_plus->select0_pair_unchecked(*G_minus, rank); })
                return G_plus->select0_pair_unchecked(*G_minus, rank);
            else
                return {G_plus->select0_unchecked(rank), G_minus->select0_unchecked(rank)};
        }

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

            if ((b && N > SIZE_MAX / b) ||
                (b < total_bits && (gap_computation.sum_of_positive_gaps > SIZE_MAX - N ||
                                    gap_computation.sum_of_negative_gaps > SIZE_MAX - N)))
                return SIZE_MAX; // This candidate cannot represent its bit vectors.
            const auto bits_to_bytes = [](size_t bits) -> size_t { return bits / 8 + (bits % 8 != 0); };
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
        static unsigned __int128 exact_total_variation(const C &S, const T min, const T max) {
            using U = std::make_unsigned_t<T>;
            const U range = static_cast<U>(max) - static_cast<U>(min);
            if (range == 0 || S.size() == 0) return 0;

            // All magnitudes are at most range. Proving the sum fits once lets
            // the common path use an ordinary, vectorizable word-sized sum.
            if (range <= SIZE_MAX / S.size()) {
                size_t variation = static_cast<U>(static_cast<U>(S[0]) - static_cast<U>(min));
                for (size_t i = 1; i < S.size(); ++i) {
                    const U current = static_cast<U>(S[i]);
                    const U previous = static_cast<U>(S[i - 1]);
                    variation += S[i] >= S[i - 1]
                        ? static_cast<U>(current - previous)
                        : static_cast<U>(previous - current);
                }
                return variation;
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
                return variation;
            }
        }

        // ---- Two-pass approximate construction ----
        //
        // APPROXIMATE_SPLIT_POINT reads S twice: one estimation pass computes
        // min, max and the total variation g(S) (with s_0 = base), and one
        // construction pass writes L, G+ and G-. The split point is
        //
        //     b = max(0, floor(log2(g / n))),
        //
        // the witness used in the proof of the paper's B*-GEF space bound, so
        // payload(b) <= n (b + 3) + floor(g / 2^b) <= n (4 + log2 max{1, g/n})
        // (rank/select support and headers excluded). Raw bit-packing replaces
        // it only when n * width is at most the matching lower bound
        // n (b + 2) + max(0, floor(g / 2^b) - n) on payload(b): it is then never
        // larger than b would be, so the guarantee holds without a statistics
        // pass. (Testing against the upper bound instead also keeps the
        // guarantee but bit-packs blocks that b stores smaller: +1.2% on WD.)

        struct RangeAndVariation {
            T minimum;
            T maximum;
            unsigned __int128 variation; // g(S) of S - minimum, with s_0 = 0
        };

        // One scan for min, max and the sum of |S[i] - S[i-1]|. Every term is
        // a plain reduction, so the loop vectorizes like the min/max scan it
        // replaces. 64-bit magnitudes are summed as two 32-bit halves, which
        // cannot overflow a 64-bit accumulator before n reaches 2^32.
        template<typename C>
        static RangeAndVariation range_and_variation(const C &S) {
            using U = std::make_unsigned_t<T>;
            using Wide = unsigned __int128;
            const size_t N = S.size();
            T minimum = S[0];
            T maximum = S[0];
            if (N > UINT32_MAX) [[unlikely]] {
                for (size_t i = 1; i < N; ++i) {
                    minimum = std::min(minimum, static_cast<T>(S[i]));
                    maximum = std::max(maximum, static_cast<T>(S[i]));
                }
                return {minimum, maximum, exact_total_variation(S, minimum, maximum)};
            }
            uint64_t low_sum = 0;
            uint64_t high_sum = 0;
            // Selects and a sign-mask absolute value instead of std::min/max
            // and a conditional difference: GCC 13 if-converts and vectorizes
            // this form (AVX2 and AVX-512), but not the conditional one.
            for (size_t i = 1; i < N; ++i) {
                const T current = S[i];
                const T previous = S[i - 1];
                minimum = current < minimum ? current : minimum;
                maximum = current > maximum ? current : maximum;
                const U sign = U(0) - static_cast<U>(current < previous);
                const uint64_t magnitude = static_cast<U>(
                    (static_cast<U>(static_cast<U>(current) - static_cast<U>(previous)) ^ sign) - sign);
                if constexpr (sizeof(T) >= sizeof(uint64_t)) {
                    low_sum += magnitude & UINT32_MAX;
                    high_sum += magnitude >> 32;
                } else {
                    low_sum += magnitude;
                }
            }
            Wide variation = (static_cast<Wide>(high_sum) << 32) + low_sum;
            variation += static_cast<U>(static_cast<U>(S[0]) - static_cast<U>(minimum));
            return {minimum, maximum, variation};
        }

        // Split point of the two-pass construction for n values of the given
        // width and total variation. Returns width for raw bit-packing.
        static uint8_t two_pass_split_point(const size_t n, const uint8_t width,
                                            const unsigned __int128 variation) {
            using Wide = unsigned __int128;
            if (width == 0 || n == 0) return width;
            // variation <= n * range, so the mean is below 2^width and the
            // split is admissible (< width).
            const uint64_t mean = static_cast<uint64_t>(variation / n);
            const uint8_t split = mean ? static_cast<uint8_t>(std::bit_width(mean) - 1) : 0;
            // sum |Gap_b(i)| > g / 2^b - n, so this never exceeds payload(split).
            const Wide scaled = variation >> split;
            const Wide lower_bound = static_cast<Wide>(n) * (split + 2u) + (scaled > n ? scaled - n : 0);
            return static_cast<Wide>(n) * width <= lower_bound ? width : split;
        }

        // Upper bounds, in bits, of G+ and G- at split b < width. With
        // P + M = sum |Gap_b(i)| <= floor(g / 2^b) + n (gaps between floors
        // differ from the scaled gaps by less than one) and P - M = last high
        // part (the high parts start from s_0 = 0), each stream holds its n
        // delimiters plus at most (bound +- last) / 2 ones.
        static std::pair<size_t, size_t> gap_stream_capacities(const size_t n, const uint8_t b,
                                                               const unsigned __int128 variation,
                                                               const size_t last_high) {
            const size_t gap_bound = static_cast<size_t>(variation >> b) + n; // < 3n
            assert(last_high <= gap_bound);
            return {n + (gap_bound + last_high) / 2, n + (gap_bound - last_high) / 2};
        }

        // Write area for the gap streams before their exact lengths are known.
        // Partition-sized requests reuse a per-thread buffer that stays in cache.
        static uint64_t* gap_scratch(const size_t words, std::unique_ptr<uint64_t[]>& owned) {
            constexpr size_t cached_words = size_t(1) << 17; // 1 MiB per thread
            if (words <= cached_words) {
                thread_local std::vector<uint64_t> cache;
                if (cache.size() < words) cache.resize(words);
                return cache.data();
            }
            owned.reset(new uint64_t[words]);
            return owned.get();
        }

        template<typename C>
        static std::pair<uint8_t, GapComputation>
        optimal_split_point_candidates(const C &S, const T min, const T max,
                                       const size_t min_b, const size_t max_b) {
            const size_t total_bits = value_width(min, max);
            const size_t count = max_b - min_b + 1;
            assert(count <= 4);
            std::array<GapComputation, 4> gap_computation{};
            using U = std::make_unsigned_t<T>;
            // min_b <= total_bits-1 for a nonconstant block, even at width 64.
            assert(min_b < sizeof(T) * 8);
            const U maximum_high = static_cast<U>(static_cast<U>(max) - static_cast<U>(min)) >> min_b;
            if (maximum_high <= (SIZE_MAX - S.size()) / S.size()) {
                // Every gap is at most maximum_high. This proves all absolute
                // sums and their delimiter counts fit without saturating adds.
                std::array<size_t, 4> variation{};
                // The first previous high part is zero. Later ones are re-read
                // from S, so no value is carried between iterations and the scan
                // vectorizes.
                const auto high = [&](size_t i) -> U {
                    return static_cast<U>(static_cast<U>(S[i]) - static_cast<U>(min)) >> min_b;
                };
                const auto accumulate = [&]<size_t FixedCount>() {
                    const size_t candidate_count = FixedCount ? FixedCount : count;
                    for (size_t j = 0; j < candidate_count; ++j) variation[j] = high(0) >> j;
                    for (size_t i = 1; i < S.size(); ++i) {
                        U current = high(i);
                        U prior = high(i - 1);
                        for (size_t j = 0; j < candidate_count; ++j) {
                            variation[j] += current >= prior ? current - prior : prior - current;
                            current >>= 1;
                            prior >>= 1;
                        }
                    }
                };
                // A full paper neighborhood has four candidates. Keep its
                // bound constant while sharing the scan with clipped edges.
                if (count == 4) accumulate.template operator()<4>();
                else accumulate.template operator()<0>();
                U previous = high(S.size() - 1);
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

        template<typename C>
        static std::pair<uint8_t, GapComputation>
        optimal_split_point(const C &S, const T min, const T max) {
            const size_t width = value_width(min, max);
            if (!width) return {0, bitpack_statistics(S.size())};
            using Wide = unsigned __int128;
            using U = std::make_unsigned_t<T>;
            const Wide variation = exact_total_variation(S, min, max);
            const uint64_t mean = static_cast<uint64_t>(variation / S.size());
            const size_t center = mean ? std::bit_width(mean) - 1 : 0;
            // Paper neighborhood: an unrounded bit optimum is among these four
            // neighbors or raw bit-packing (which the candidate helper includes).
            const size_t begin = center ? center - 1 : 0;
            const size_t end = std::min(width - 1, center + 2);
            auto best = optimal_split_point_candidates(S, min, max, begin, end);
            size_t best_bytes = evaluate_space(best.second, best.first, width, S.size());
            const auto ceil_bytes = [](Wide bits) { return bits / 8 + (bits % 8 != 0); };
            const Wide n = S.size();
            // The theorem concerns bits; certify every omitted split against
            // the existing separately rounded byte objective, including ties.
            for (size_t k = 0; k < width; ++k) {
                if (k >= begin && k <= end) continue;
                const Wide scaled = variation >> k;
                const Wide lower_gap = scaled >= n ? scaled - n + 1 : 0;
                const Wide payload = std::max(ceil_bytes(n * (k + 2) + lower_gap),
                                               ceil_bytes(n * k) + 2 * ceil_bytes(n));
                const Wide lower_bytes = sizeof(T) + sizeof(uint8_t) * 2 + payload;
                if (lower_bytes > best_bytes || (lower_bytes == best_bytes && k > best.first)) continue;

                // Rare rounding ambiguities need an exact single-candidate scan.
                // Wide arithmetic avoids saturating a gap sum into a false cost.
                Wide absolute = 0;
                U previous = 0;
                for (const T value : S) {
                    const U current = static_cast<U>(static_cast<U>(value) - static_cast<U>(min)) >> k;
                    absolute += current >= previous ? current - previous : previous - current;
                    previous = current;
                }
                const Wide negative = (absolute - previous) / 2;
                const Wide positive = negative + previous;
                const Wide bytes = sizeof(T) + sizeof(uint8_t) * 2 + ceil_bytes(n * k) +
                                   ceil_bytes(n + positive) + ceil_bytes(n + negative);
                if (bytes < best_bytes || (bytes == best_bytes && k < best.first)) {
                    if (positive + n > SIZE_MAX || negative + n > SIZE_MAX || n * k > SIZE_MAX) continue;
                    GapComputation stats{};
                    stats.sum_of_positive_gaps = static_cast<size_t>(positive);
                    stats.sum_of_negative_gaps = static_cast<size_t>(negative);
                    best = {static_cast<uint8_t>(k), stats};
                    best_bytes = static_cast<size_t>(bytes);
                }
            }
            return best;
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
            template<typename Position>
            void read(Position* out, size_t count) {
                static_assert(std::is_same_v<Position, size_t> || std::is_same_v<Position, uint32_t>);
                size_t i = 0;
                while (i < count) {
                    while (zeros_ == 0) zeros_ = ~data_[++word_];
                    const size_t base_position = word_ * 64;
                    // Expand a full delimiter word without per-value bit scans.
                    if (zeros_ == UINT64_MAX && count - i >= 64) {
                        for (size_t offset = 0; offset < 64; ++offset)
                            out[i + offset] = static_cast<Position>(base_position + offset);
                        i += 64;
                        zeros_ = 0;
                        continue;
                    }
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD) && SIZE_MAX == UINT64_MAX
                    if constexpr (!GapBitVectorType::reverse_bit_order) {
#if defined(__AVX512F__)
                        if constexpr (sizeof(Position) == 4) {
                            if (count - i >= 64) {
                                const __m512i offsets = _mm512_setr_epi32(
                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                                __m512i positions = _mm512_add_epi32(offsets, _mm512_set1_epi32(
                                    static_cast<int32_t>(base_position)));
                                const auto emit_packet = [&](unsigned shift) __attribute__((always_inline)) {
                                    const unsigned mask = static_cast<unsigned>((zeros_ >> shift) & 65535);
                                    if (shift) positions = _mm512_add_epi32(positions, _mm512_set1_epi32(16));
                                    _mm512_mask_compressstoreu_epi32(out + i, static_cast<__mmask16>(mask), positions);
                                    i += std::popcount(mask);
                                };
                                // Four packets write at most the 64 available slots.
                                // zeros_ already excludes delimiters before the cursor.
                                [&]<size_t... Packets>(std::index_sequence<Packets...>) __attribute__((always_inline)) {
                                    (emit_packet(16 * Packets), ...);
                                }(std::make_index_sequence<4>{});
                                zeros_ = 0;
                                continue;
                            }
                        }
#endif
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
                            if constexpr (sizeof(Position) == 4) {
                                const __m256i base = _mm256_set1_epi32(std::bit_cast<int32_t>(
                                    static_cast<uint32_t>(base_position + shift)));
                                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i),
                                    _mm256_add_epi32(base, _mm256_cvtepu8_epi32(bytes)));
                            } else {
                                const __m256i base = _mm256_set1_epi64x(
                                    std::bit_cast<int64_t>(uint64_t(base_position + shift)));
                                const __m256i first = _mm256_add_epi64(base, _mm256_cvtepu8_epi64(bytes));
                                const __m256i second = _mm256_add_epi64(base,
                                    _mm256_cvtepu8_epi64(_mm_srli_si128(bytes, 4)));
                                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), first);
                                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 4), second);
                            }
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
                            out[i++] = static_cast<Position>(base_position + bit);
                            zeros_ ^= uint64_t(1) << (63 - bit);
                        } else {
                            out[i++] = static_cast<Position>(base_position + std::countr_zero(zeros_));
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
                const uint64_t bit_position = static_cast<uint64_t>(index) * SPLIT_POINT;
                const size_t word = static_cast<size_t>(bit_position >> 6);
                const unsigned offset = bit_position & 63;
                uint64_t value = data[word] >> offset;
                if constexpr (64 % SPLIT_POINT != 0) {
                    if (offset + SPLIT_POINT > 64)
                        value |= data[word + 1] << (64 - offset);
                }
                return value & mask;
            }
        }

#if defined(__AVX2__) && defined(__AVX512F__) && !defined(GEF_DISABLE_SIMD) && SIZE_MAX == UINT64_MAX
        template<uint8_t SPLIT_POINT, size_t INDEX>
        static void reconstruct_low8(const uint64_t* words, const uint32_t* plus_positions,
                                     const uint32_t* minus_positions, __m512i base_vector, T* output) {
            static_assert(sizeof(T) == 8 && sizeof(size_t) == 8 && SPLIT_POINT >= 1 &&
                          SPLIT_POINT <= 16 && INDEX % 8 == 0 && INDEX <= 56);
            constexpr size_t word = (INDEX * SPLIT_POINT) / 64;
            constexpr unsigned bit = (INDEX * SPLIT_POINT) % 64;
            constexpr unsigned window_values = SPLIT_POINT <= 8 ? 8 : 4;
            uint64_t window = words[word] >> bit;
            if constexpr (bit + window_values * SPLIT_POINT > 64)
                window |= words[word + 1] << (64 - bit);
            __m512i source = _mm512_set1_epi64(std::bit_cast<int64_t>(window));
            if constexpr (SPLIT_POINT > 8) {
                // Each half stays inside its bounded four-field low window.
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
            const __m512i lows = _mm512_and_si512(_mm512_srlv_epi64(source, shifts), mask);
            const __m256i plus = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(plus_positions + INDEX));
            const __m256i minus = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(minus_positions + INDEX));
            // Parent's BOTH-stream INT32 guard also bounds negative differences.
            const __m512i high = _mm512_slli_epi64(
                _mm512_cvtepi32_epi64(_mm256_sub_epi32(plus, minus)), SPLIT_POINT);
            _mm512_storeu_si512(output + INDEX,
                _mm512_add_epi64(base_vector, _mm512_or_si512(high, lows)));
        }
#endif

        // At an element index divisible by 64, the next 64 low parts occupy
        // exactly SPLIT_POINT words. Constant indices remove per-value offset
        // and word-crossing decisions without changing the packed representation.
        template<uint8_t SPLIT_POINT, typename Position, size_t... I>
        static void reconstruct_aligned64(const uint64_t* low_data,
                                          const Position* plus_positions,
                                          const Position* minus_positions,
                                          std::make_unsigned_t<T> base_u,
                                          T* output,
                                          std::index_sequence<I...>) {
            static_assert(SPLIT_POINT > 0 && SPLIT_POINT < sizeof(T) * 8);
            static_assert(sizeof...(I) == 64);
            using U = std::make_unsigned_t<T>;
#if defined(__AVX2__) && defined(__AVX512F__) && !defined(GEF_DISABLE_SIMD) && SIZE_MAX == UINT64_MAX
            if constexpr (sizeof(T) == 8 && sizeof(size_t) == 8 && sizeof(Position) == 4 &&
                          !GapBitVectorType::reverse_bit_order && SPLIT_POINT <= 16) {
                const __m512i base_vector = _mm512_set1_epi64(std::bit_cast<int64_t>(base_u));
                ([&] {
                    if constexpr (I % 8 == 0)
                        reconstruct_low8<SPLIT_POINT, I>(low_data, plus_positions, minus_positions, base_vector, output);
                }(), ...);
                return;
            }
#endif
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
                        __m256i difference;
                        if constexpr (sizeof(Position) == 4) {
                            const __m128i plus = _mm_loadu_si128(reinterpret_cast<const __m128i*>(plus_positions + I));
                            const __m128i minus = _mm_loadu_si128(reinterpret_cast<const __m128i*>(minus_positions + I));
                            // Both absolute positions fit [0, INT32_MAX], so
                            // their signed difference fits int32, even for
                            // loaded streams whose plus-minus is negative.
                            difference = _mm256_cvtepi32_epi64(_mm_sub_epi32(plus, minus));
                        } else {
                            const __m256i plus = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(plus_positions + I));
                            const __m256i minus = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(minus_positions + I));
                            difference = _mm256_sub_epi64(plus, minus);
                        }
                        const __m256i high = _mm256_slli_epi64(difference, SPLIT_POINT);
                        const __m256i values = _mm256_add_epi64(base_vector, _mm256_or_si256(high, lows));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + I), values);
                    }
                }(), ...);
            } else
#endif
            {
                ([&] {
                    const U high = static_cast<U>(static_cast<size_t>(plus_positions[I]) -
                                                  static_cast<size_t>(minus_positions[I])) << SPLIT_POINT;
                    const U low = static_cast<U>(read_low<SPLIT_POINT>(low_data, I));
                    output[I] = static_cast<T>(base_u + (high | low));
                }(), ...);
            }
        }

        template<uint8_t SPLIT_POINT, typename Position>
        size_t decode_position_ranges(size_t startIndex, size_t count, T* output) const {
            constexpr size_t MAX_BATCH = sizeof(Position) < sizeof(size_t) ? 2048 : 256;
            using U = std::make_unsigned_t<T>;
            const U base_u = static_cast<U>(base);
            const uint64_t* low_data = L.data();
            const size_t plus_start = startIndex ? gap_select0(0, startIndex) + 1 : 0;
            const size_t minus_start = startIndex ? gap_select0(1, startIndex) + 1 : 0;
            ZeroPositionReader plus(G_plus->raw_data_ptr(), plus_start);
            ZeroPositionReader minus(G_minus->raw_data_ptr(), minus_start);
            alignas(64) Position plus_positions[MAX_BATCH];
            alignas(64) Position minus_positions[MAX_BATCH];
            // End the first batch on a low-block boundary; later batches stay aligned.
            const size_t first_batch = MAX_BATCH -
                ((SPLIT_POINT > 0 && SPLIT_POINT < sizeof(T) * 8) ? (startIndex & 63) : 0);
            for (size_t produced = 0; produced < count;) {
                const size_t batch = std::min(count - produced, produced == 0 ? first_batch : MAX_BATCH);
                plus.read(plus_positions, batch);
                minus.read(minus_positions, batch);
                const auto reconstruct_scalar = [&](size_t i) {
                    U high = 0;
                    if constexpr (SPLIT_POINT < sizeof(T) * 8)
                        high = static_cast<U>(static_cast<size_t>(plus_positions[i]) -
                                              static_cast<size_t>(minus_positions[i])) << SPLIT_POINT;
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

        template<uint8_t SPLIT_POINT>
        size_t get_elements_worker(size_t startIndex, size_t count, T* output) const {
            if (count == 0 || startIndex >= m_num_elements) [[unlikely]] return 0;
            count = std::min(count, m_num_elements - startIndex);
            using U = std::make_unsigned_t<T>;
            if (h == 0) {
                if constexpr (SPLIT_POINT == 0) {
                    std::fill_n(output, count, base);
                } else {
                    const U base_u = static_cast<U>(base);
                    const uint64_t* low_data = L.data();
                    for (size_t i = 0; i < count; ++i)
                        output[i] = static_cast<T>(base_u + static_cast<U>(read_low<SPLIT_POINT>(low_data, startIndex + i)));
                }
                return count;
            }
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD) && SIZE_MAX == UINT64_MAX
            if constexpr (sizeof(T) == 8 && !GapBitVectorType::reverse_bit_order) {
                // Bound absolute bit positions, including the last word's
                // speculative byte slots; partition element count is not enough.
                if (G_plus->size() <= INT32_MAX && G_minus->size() <= INT32_MAX)
                    return decode_position_ranges<SPLIT_POINT, uint32_t>(startIndex, count, output);
            }
#endif
            return decode_position_ranges<SPLIT_POINT, size_t>(startIndex, count, output);
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
                G_minus = std::make_unique<GapBitVectorType>(*other.G_minus);
                enable_random_access();
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
            swap(first.zero_select, second.zero_select);
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
              zero_select(std::move(other.zero_select)),
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
                zero_select = std::move(other.zero_select);
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

            // The approximate strategy folds g(S) into the estimation pass and
            // then needs only the construction pass (see two_pass_split_point).
            const bool two_pass = !parameters && strategy == APPROXIMATE_SPLIT_POINT;
            unsigned __int128 variation = 0;
            T minimum = *S.begin();
            T maximum = minimum;
            if (two_pass) {
                const auto estimate = range_and_variation(S);
                minimum = estimate.minimum;
                maximum = estimate.maximum;
                variation = estimate.variation;
            } else {
                // Independent reductions avoid minmax_element's pairwise branch
                // and let the compiler vectorize contiguous input scans.
                for (const auto value : S) {
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
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
                    b = two_pass_split_point(N, total_bits, variation);
                    // MSB-first backends set ones in place and need exact
                    // stream lengths up front: one statistics scan at b.
                    if constexpr (GapBitVectorType::reverse_bit_order) {
                        if (b < total_bits)
                            gap_computation = variation_of_shifted_vec(S, base, max_val, b, ExceptionRule::None);
                    }
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
                    using U = std::make_unsigned_t<T>;
                    PackedBitWriter low_writer(L.data());
                    for (size_t i = 0; i < N; ++i)
                        low_writer.append(static_cast<U>(static_cast<U>(S[i]) - static_cast<U>(base)), b);
                    low_writer.finish();
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

            using U = std::make_unsigned_t<T>;

            // Two-pass construction writes the gap streams into scratch sized
            // by their proven bounds; exact vectors are filled once the
            // lengths are known. Other paths allocate exact vectors up front.
            const bool bounded_streams = two_pass && !GapBitVectorType::reverse_bit_order;
            std::unique_ptr<uint64_t[]> owned_scratch;
            uint64_t* plus_scratch = nullptr;
            uint64_t* minus_scratch = nullptr;
            [[maybe_unused]] size_t plus_capacity = 0, minus_capacity = 0;
            size_t plus_words = 0, minus_words = 0;
            if (bounded_streams) {
                const size_t last_high = static_cast<size_t>(
                    static_cast<U>(static_cast<U>(S[N - 1]) - static_cast<U>(base)) >> b);
                std::tie(plus_capacity, minus_capacity) =
                    gap_stream_capacities(N, b, variation, last_high);
                plus_words = plus_capacity / 64 + 1;
                minus_words = minus_capacity / 64 + 1;
                plus_scratch = gap_scratch(plus_words + minus_words, owned_scratch);
                minus_scratch = plus_scratch + plus_words;
            } else {
                const size_t g_plus_bits = gap_computation.sum_of_positive_gaps + N;
                const size_t g_minus_bits = gap_computation.sum_of_negative_gaps + N;

                G_plus = std::make_unique<GapBitVectorType>(g_plus_bits);
                G_minus = std::make_unique<GapBitVectorType>(g_minus_bits);
            }

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
            const U low_mask = b ? (U(~U(0)) >> (sizeof(T) * 8 - b)) : U(0);
            
            PackedBitWriter low_writer(L.data());
            if constexpr (GapBitVectorType::reverse_bit_order) {
                // MSB-first gap backends keep the generic writer, which sets
                // ones in place in the zero-initialized vectors.
                FastBitWriter<true> plus_writer(G_plus->raw_data_ptr());
                FastBitWriter<true> minus_writer(G_minus->raw_data_ptr());
                U lastHighBits = 0;
                for (size_t i = 0; i < N; ++i) {
                    const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
                    if (b > 0) low_writer.append(static_cast<uint64_t>(element_u & low_mask), b);
                    const U currentHighBits = element_u >> b;
                    if (currentHighBits > lastHighBits)
                        plus_writer.set_ones_range(static_cast<uint64_t>(currentHighBits - lastHighBits));
                    else if (currentHighBits < lastHighBits)
                        minus_writer.set_ones_range(static_cast<uint64_t>(lastHighBits - currentHighBits));
                    plus_writer.skip_zeros(1);
                    minus_writer.skip_zeros(1);
                    lastHighBits = currentHighBits;
                }
                low_writer.finish();
            } else if (bounded_streams) {
                // Both streams are all ones except one delimiting zero per
                // element. With R_i the sum of the rises up to element i, the
                // i-th zero of G+ is at i + R_i and, since the rises minus the
                // falls telescope to the current high part c_i, the i-th zero of
                // G- is at i + R_i - c_i. The scratch is filled with ones up to
                // the proven capacity; each element then clears its zero in a
                // register copy of the word holding it and stores that word.
                // The store is unconditional and the word is reset with a
                // select, so no branch depends on the gaps (a mispredicted
                // word-boundary branch per stream was the cost of the writers).
                std::fill_n(plus_scratch, plus_words + minus_words, ~uint64_t(0));
                const auto populate = [&](auto with_lows) {
                    U last = 0;
                    uint64_t rises = 0;
                    uint64_t plus_word = 0, minus_word = 0;
                    uint64_t plus_bits = ~uint64_t(0), minus_bits = ~uint64_t(0);
                    for (size_t i = 0; i < N; ++i) {
                        const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
                        if constexpr (with_lows) low_writer.append(static_cast<uint64_t>(element_u & low_mask), b);
                        const U current = element_u >> b;
                        rises += static_cast<uint64_t>(current > last ? current - last : U(0));
                        last = current;
                        const uint64_t plus_zero = i + rises;
                        const uint64_t minus_zero = plus_zero - static_cast<uint64_t>(current);
                        const uint64_t plus_index = plus_zero >> 6, minus_index = minus_zero >> 6;
                        plus_bits = plus_index == plus_word ? plus_bits : ~uint64_t(0);
                        minus_bits = minus_index == minus_word ? minus_bits : ~uint64_t(0);
                        plus_bits &= ~(uint64_t(1) << (plus_zero & 63));
                        minus_bits &= ~(uint64_t(1) << (minus_zero & 63));
                        plus_scratch[plus_index] = plus_bits;
                        minus_scratch[minus_index] = minus_bits;
                        plus_word = plus_index;
                        minus_word = minus_index;
                    }
                    return std::pair<uint64_t, uint64_t>{N + rises, N + rises - static_cast<uint64_t>(last)};
                };
                const auto [plus_length, minus_length] =
                    b > 0 ? populate(std::true_type{}) : populate(std::false_type{});
                low_writer.finish();
                assert(plus_length <= plus_capacity && minus_length <= minus_capacity);
                // Exact-length vectors with zero-filled tails, as allocated by
                // the other paths: only the used bits of the last word are kept.
                const auto materialize = [](uint64_t* words, size_t bits) {
                    if (bits & 63) words[bits >> 6] &= (uint64_t(1) << (bits & 63)) - 1;
                    auto vector = std::make_unique<GapBitVectorType>(bits);
                    std::memcpy(vector->raw_data_ptr(), words, ((bits + 63) / 64) * sizeof(uint64_t));
                    return vector;
                };
                G_plus = materialize(plus_scratch, plus_length);
                G_minus = materialize(minus_scratch, minus_length);
            } else {
            // Register writers store each word once it is full. Rise and fall
            // are computed without a branch on the sign of the gap.
            PackedBitWriter plus_writer(G_plus->raw_data_ptr());
            PackedBitWriter minus_writer(G_minus->raw_data_ptr());
            const auto populate = [&](auto with_lows) {
                U lastHighBits = 0;
                for (size_t i = 0; i < N; ++i) {
                    const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
                    if constexpr (with_lows) low_writer.append(static_cast<uint64_t>(element_u & low_mask), b);
                    // G+ holds the unary code of each rise, G- of each fall.
                    const U currentHighBits = element_u >> b;
                    const U top = std::max(currentHighBits, lastHighBits);
                    plus_writer.append_unary(top - lastHighBits);
                    minus_writer.append_unary(top - currentHighBits);
                    lastHighBits = currentHighBits;
                }
            };
            if (b > 0) populate(std::true_type{});
            else populate(std::false_type{});
            plus_writer.finish();
            minus_writer.finish();
            low_writer.finish();
            }
            // ===== END POPULATION PHASE =====

            // Enable select0 support (rank not needed, operator[] only uses select0)
            enable_random_access();

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
            using U = std::make_unsigned_t<T>;
            const U low = static_cast<U>(read_low_unchecked(index));
            if (h == 0) [[unlikely]]
                return static_cast<T>(static_cast<U>(base) + low);

            const size_t zero_rank = index + 1;
            const auto positions = zero_select.enabled() ? zero_select.select0_pair(zero_rank)
                                                         : gap_select0_pair_general(zero_rank);
            const size_t pos_gaps = positions.first - zero_rank;
            const size_t neg_gaps = positions.second - zero_rank;

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
                G_minus = std::make_unique<GapBitVectorType>(GapBitVectorType::load(ifs));
                enable_random_access();
            } else {
                G_plus = nullptr;
                G_minus = nullptr;
                zero_select.clear();
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
                (G_minus ? G_minus->allocated_size_in_bytes() : 0) +
                zero_select.allocated_size_in_bytes();
        }

        [[nodiscard]] size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (h > 0) {
                total_bytes += G_plus->size_in_bytes();
                total_bytes += G_minus->size_in_bytes();
                total_bytes += zero_select.size_in_bytes();
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
