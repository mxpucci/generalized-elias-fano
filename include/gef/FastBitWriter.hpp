//
// Shared fast bit writer utility for GEF implementations.
//

#ifndef GEF_FAST_BIT_WRITER_HPP
#define GEF_FAST_BIT_WRITER_HPP

#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
#include <immintrin.h>
#elif defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
#include <arm_neon.h>
#endif

namespace gef {
    /**
     * @brief Fast inline bit writer for hot path optimization
     * @tparam ReverseOrder If true, writes bits from MSB (63) down to LSB (0) within a word.
     * (Required for pasta::bit_vector compatibility).
     */
    template<bool ReverseOrder = false>
    class FastBitWriter {
    private:
        uint64_t* data_;
        size_t pos_;

    public:
        explicit FastBitWriter(uint64_t* data, size_t start_pos = 0)
            : data_(data), pos_(start_pos) {}

        /**
         * @brief Set a single bit to 0 (optimized for terminator pattern)
         */
        __attribute__((always_inline)) inline void set_zero() {
            const size_t word_idx = pos_ >> 6;
            const uint32_t bit_offset = static_cast<uint32_t>(pos_ & 63);
            
            if constexpr (ReverseOrder) {
                // Pasta/Reverse: Index 0 is MSB (bit 63)
                data_[word_idx] &= ~(1ULL << (63 - bit_offset));
            } else {
                // Standard: Index 0 is LSB (bit 0)
                data_[word_idx] &= ~(1ULL << bit_offset);
            }
            ++pos_;
        }

        /**
         * @brief Set a range of bits to 1 (optimized for all gap sizes)
         */
        __attribute__((always_inline)) inline void set_ones_range(uint64_t count) {
            if (count == 0) return;

            const size_t end_pos = pos_ + count - 1;
            const size_t start_word = pos_ >> 6;
            const size_t end_word = end_pos >> 6;
            const uint32_t start_bit = static_cast<uint32_t>(pos_ & 63);
            const uint32_t end_bit = static_cast<uint32_t>(end_pos & 63);

            if (start_word == end_word) {
                // Single word case
                uint64_t mask = (~0ULL >> (64 - count));
                
                if constexpr (ReverseOrder) {
                    // Shift to the "left" (High bits) based on start_bit
                    mask <<= (64 - count - start_bit);
                } else {
                    // Shift to "right" (Low bits) based on start_bit
                    mask <<= start_bit;
                }
                
                data_[start_word] |= mask;
            } else {
                // Spans multiple words
                
                // --- HEAD ---
                if constexpr (ReverseOrder) {
                    // Reverse: Fill from start_bit "down" to 0. 
                    // This creates a mask of 1s at the BOTTOM of the word.
                    data_[start_word] |= (~0ULL >> start_bit);
                } else {
                    // Standard: Fill from start_bit "up" to 63.
                    data_[start_word] |= (~0ULL << start_bit);
                }

                const size_t num_full_words = end_word - start_word - 1;
                if (num_full_words > 0) {
                    uint64_t* body_start = data_ + start_word + 1;
                    
                    // SIMD Body (Endian agnostic for 0xFFFFFF...)
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
                    __m256i ones = _mm256_set1_epi64x(~0ULL);
                    size_t w = 0;
                    for (; w + 4 <= num_full_words; w += 4) {
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(body_start + w), ones);
                    }
                    for (; w < num_full_words; ++w) body_start[w] = ~0ULL;
#elif defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
                    uint64x2_t ones = vdupq_n_u64(~0ULL);
                    size_t w = 0;
                    for (; w + 2 <= num_full_words; w += 2) {
                        vst1q_u64(body_start + w, ones);
                    }
                    for (; w < num_full_words; ++w) body_start[w] = ~0ULL;
#else
                    // Unrolled scalar
                    size_t w = 0;
                    for (; w + 4 <= num_full_words; w += 4) {
                        body_start[w] = ~0ULL; body_start[w+1] = ~0ULL;
                        body_start[w+2] = ~0ULL; body_start[w+3] = ~0ULL;
                    }
                    for (; w < num_full_words; ++w) body_start[w] = ~0ULL;
#endif
                }

                // --- TAIL ---
                if constexpr (ReverseOrder) {
                    // Reverse: Fill from MSB (63) down to end_bit (inclusive).
                    // This creates a mask of 1s at the TOP of the word.
                    // We want (end_bit + 1) bits at the top.
                    data_[end_word] |= (~0ULL << (63 - end_bit));
                } else {
                    // Standard: Fill from 0 up to end_bit.
                    data_[end_word] |= (~0ULL >> (63 - end_bit));
                }
            }

            pos_ += count;
        }

        size_t position() const { return pos_; }
    };
} // namespace gef

#endif // GEF_FAST_BIT_WRITER_HPP