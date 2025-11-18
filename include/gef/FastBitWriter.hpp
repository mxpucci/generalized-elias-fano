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
     * Eliminates virtual function call overhead during construction by directly
     * manipulating bit vector memory. Optimized for small gap sequences.
     */
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
            data_[word_idx] &= ~(1ULL << bit_offset);
            ++pos_;
        }

        /**
         * @brief Set a range of bits to 1 (optimized for all gap sizes)
         * @param count Number of consecutive bits to set to 1
         *
         * Optimized for:
         * - Small gaps (0-8 bits): ~90% of cases in high-entropy data
         * - Large gaps (100s-1000s bits): common when split point is near 0
         */
        __attribute__((always_inline)) inline void set_ones_range(uint64_t count) {
            if (count == 0) return;

            // Unified path using word-level operations.
            // This eliminates the loop overhead for small counts and uses efficient masking.
            const size_t end_pos = pos_ + count - 1;
            const size_t start_word = pos_ >> 6;
            const size_t end_word = end_pos >> 6;
            const uint32_t start_bit = static_cast<uint32_t>(pos_ & 63);
            const uint32_t end_bit = static_cast<uint32_t>(end_pos & 63);

            if (start_word == end_word) {
                // All bits in same word - single mask operation
                // (~0ULL >> (64 - count)) creates a mask of 'count' ones at the LSBs.
                // We then shift it to the correct position.
                // This handles count=64 correctly if start_bit=0.
                const uint64_t mask = (~0ULL >> (64 - count)) << start_bit;
                data_[start_word] |= mask;
            } else {
                // Spans multiple words
                // Head: set bits from start_bit to 63
                data_[start_word] |= (~0ULL << start_bit);

                const size_t num_full_words = end_word - start_word - 1;
                if (num_full_words > 0) {
                    uint64_t* body_start = data_ + start_word + 1;

#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
                    // AVX2 path: write 4 words (32 bytes) at a time
                    __m256i ones = _mm256_set1_epi64x(~0ULL);
                    size_t w = 0;
                    // Process 4 words at a time
                    for (; w + 4 <= num_full_words; w += 4) {
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(body_start + w), ones);
                    }
                    // Handle remaining 0-3 words
                    for (; w < num_full_words; ++w) {
                        body_start[w] = ~0ULL;
                    }
#elif defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
                    // NEON path: write 2 words (16 bytes) at a time
                    uint64x2_t ones = vdupq_n_u64(~0ULL);
                    size_t w = 0;
                    // Process 2 words at a time
                    for (; w + 2 <= num_full_words; w += 2) {
                        vst1q_u64(body_start + w, ones);
                    }
                    // Handle remaining 0-1 word
                    for (; w < num_full_words; ++w) {
                        body_start[w] = ~0ULL;
                    }
#else
                    // Scalar path with loop unrolling for better performance
                    size_t w = 0;
                    // Unroll by 4 for better ILP (instruction-level parallelism)
                    for (; w + 4 <= num_full_words; w += 4) {
                        body_start[w] = ~0ULL;
                        body_start[w + 1] = ~0ULL;
                        body_start[w + 2] = ~0ULL;
                        body_start[w + 3] = ~0ULL;
                    }
                    // Handle remaining 0-3 words
                    for (; w < num_full_words; ++w) {
                        body_start[w] = ~0ULL;
                    }
#endif
                }

                // Tail: set bits from 0 to end_bit
                // (~0ULL >> (63 - end_bit)) creates a mask of (end_bit + 1) ones at LSBs
                const uint64_t tail_mask = ~0ULL >> (63 - end_bit);
                data_[end_word] |= tail_mask;
            }

            pos_ += count;
        }

        size_t position() const { return pos_; }
    };
} // namespace gef

#endif // GEF_FAST_BIT_WRITER_HPP


