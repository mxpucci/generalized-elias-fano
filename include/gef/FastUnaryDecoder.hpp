//
// Fast unary-decoder utility to reconstruct run-length encoded gaps.
// Refactored for High-Performance Hot Path.
// OPTIMIZED: Register caching, localized state, and reduced branching.
//

#ifndef GEF_FAST_UNARY_DECODER_HPP
#define GEF_FAST_UNARY_DECODER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// Includes for SIMD/Bit manipulation intrinsics
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(__BMI1__)
#include <immintrin.h>
#define RESET_LOWEST_BIT(x) _blsr_u64(x)
#else
#define RESET_LOWEST_BIT(x) ((x) & ((x) - 1))
#endif

namespace gef {

    /**
     * @brief Fast decoder for Unary codes (gaps).
     * @tparam ReverseOrder 
     * false (Default/SDSL/Pasta): Bits read LSB -> MSB.
     * true: Bits read MSB -> LSB.
     */
    template<bool ReverseOrder = false>
    class FastUnaryDecoder {
    private:
        const uint64_t* data_;
        size_t total_bits_;
        size_t pos_;
        size_t pending_gap_; 

        // Cross-platform TZCNT (Count Trailing Zeros)
        static inline uint32_t fast_ctz(uint64_t x) {
#if defined(_MSC_VER)
            unsigned long index;
            _BitScanForward64(&index, x);
            return (uint32_t)index;
#else
            return (uint32_t)__builtin_ctzll(x);
#endif
        }

        // Cross-platform LZCNT (Count Leading Zeros)
        static inline uint32_t fast_clz(uint64_t x) {
#if defined(_MSC_VER)
            unsigned long index;
            _BitScanReverse64(&index, x);
            return 63 - (uint32_t)index;
#else
            return (uint32_t)__builtin_clzll(x);
#endif
        }

    public:
        FastUnaryDecoder(const uint64_t* data, size_t total_bits, size_t start_pos = 0)
            : data_(data),
              total_bits_(total_bits),
              pos_(std::min(start_pos, total_bits)),
              pending_gap_(0) {}

        // Single decode remains available, but next_batch is preferred
        inline uint64_t next() {
            uint32_t tmp;
            if (next_batch(&tmp, 1) == 1) return tmp;
            return 0;
        }

        /**
        * @brief Prefetches data ahead of the current position.
        * Useful when doing heavy processing between batches.
        * @param cache_lines_ahead How many 64-byte cache lines to skip.
        */
        inline void prefetch(size_t cache_lines_ahead = 4) const {
            #if defined(__GNUC__) || defined(__clang__)
                // pos_ is in bits. pos_ >> 6 is uint64_t index.
                // We want to prefetch 'cache_lines_ahead' * 64 bytes (8 uint64_t) ahead.
                const size_t current_word_idx = pos_ >> 6;
                const size_t lookahead_word_idx = current_word_idx + (cache_lines_ahead * 8);
                
                // Check bounds roughly to avoid prefetching into unmapped pages (rare but possible)
                if (lookahead_word_idx * 64 < total_bits_) {
                    // 0 = Read access, 1 = Low temporal locality (we read once and move on)
                    __builtin_prefetch(data_ + lookahead_word_idx, 0, 1);
                }
            #endif
            }

        /**
         * @brief Decodes a batch of gaps.
         * Optimized: Loads 64-bit words into registers and drains them completely
         * before advancing memory pointers.
         */
        size_t next_batch(uint32_t* out, size_t max_batch) {
            // Localize state to registers to avoid pointer aliasing
            size_t local_pos = pos_;
            size_t local_pending = pending_gap_;
            size_t produced = 0;
            const size_t limit = total_bits_;
            const uint64_t* ptr = data_;

            // If we are out of data, return immediately
            if (local_pos >= limit) return 0;

            // Calculate current word index and bit offset
            size_t word_idx = local_pos >> 6;
            uint32_t bit_off = static_cast<uint32_t>(local_pos & 63);

            // Pre-load current word
            uint64_t curr_word = ptr[word_idx];

            while (produced < max_batch && local_pos < limit) {
                // Determine how many bits are actually valid in this word 
                // (usually 64, unless we are at the very last word of the stream)
                // We use a branchless min for the tail case logic effectively.
                size_t bits_available_in_word = 64 - bit_off;
                if (local_pos + bits_available_in_word > limit) {
                    bits_available_in_word = limit - local_pos;
                }

                if constexpr (ReverseOrder) {
                    // --- REVERSE (MSB -> LSB) ---
                    
                    // Invert: We look for 0s (terminators), so we invert 1s to 0s.
                    // Now we are looking for the first '1'.
                    uint64_t inv = ~curr_word;

                    // Mask out bits we have already processed (higher than bit_off)
                    // If bit_off is 0, we keep everything. If 1, we clear MSB.
                    // (~0ULL >> bit_off) keeps the LOWER (64-bit_off) bits.
                    if (bit_off > 0) inv &= (~0ULL >> bit_off);
                    
                    // We also need to mask out bits BEYOND the valid stream end (if at last word)
                    // If bits_available < (64 - bit_off), we have tail garbage.
                    // We only want to keep the top 'bits_available' bits relative to bit_off.
                    // This is complex in reverse. Simplification:
                    // Just set the invalid low bits to 0 so they don't trigger CLZ.
                    size_t invalid_tail = 64 - (bit_off + bits_available_in_word);
                    if (invalid_tail > 0) inv &= (~0ULL << invalid_tail);

                    // Inner tight loop: process all gaps in this word
                    while (produced < max_batch && inv != 0) {
                        uint32_t lz = fast_clz(inv); // Index of first '1' (which was '0')
                        
                        // Gap size = distance from current offset to the terminator
                        uint32_t run = lz - bit_off;
                        
                        out[produced++] = static_cast<uint32_t>(local_pending + run);
                        local_pending = 0;

                        // Advance
                        uint32_t consumed = run + 1;
                        local_pos += consumed;
                        bit_off += consumed;
                        
                        // Clear the found bit in 'inv' so we find the next one
                        // clear bit at index (63 - lz)
                        inv &= ~(1ULL << (63 - lz));
                    }
                    
                    // If we finished the word (or ran out of valid bits)
                    if (inv == 0) {
                        // How many bits were left in this word that were all 1s (unary)?
                        // If we are here, either we consumed everything, OR the rest of the word is 1s.
                        // Re-calculate remaining valid bits based on new local_pos
                        size_t bits_left_in_word = (word_idx + 1) * 64 - local_pos;
                        if (local_pos >= limit) bits_left_in_word = 0; // Clamp
                        else if (local_pos + bits_left_in_word > limit) bits_left_in_word = limit - local_pos;

                        if (bits_left_in_word > 0) {
                            local_pending += bits_left_in_word;
                            local_pos += bits_left_in_word;
                        }
                        
                        // Move to next word
                        word_idx++;
                        bit_off = 0;
                        if (local_pos < limit) curr_word = ptr[word_idx];
                    }

                } else {
                    // --- STANDARD (LSB -> MSB) ---
                    
                    // Invert: Unary 1s become 0s. Terminator 0 becomes 1.
                    uint64_t inv = ~curr_word;

                    // Mask out previously processed bits (lower than bit_off).
                    // We set them to 0 so ctz skips them.
                    if (bit_off > 0) inv &= (~0ULL << bit_off);

                    // Mask out garbage beyond the total stream size (if last word)
                    // If (bit_off + bits_available) < 64, we need to clear top bits.
                    if ((bit_off + bits_available_in_word) < 64) {
                        inv &= (~0ULL >> (64 - (bit_off + bits_available_in_word)));
                    }

                    // TIGHT LOOP: Drain the register
                    while (produced < max_batch && inv != 0) {
                        uint32_t tz = fast_ctz(inv); // absolute bit index of terminator

                        // Run length is distance from current offset
                        uint32_t run = tz - bit_off;

                        out[produced++] = static_cast<uint32_t>(local_pending + run);
                        local_pending = 0;

                        // Advance
                        uint32_t consumed = run + 1;
                        local_pos += consumed;
                        bit_off += consumed;

                        // Clear the bit we just found so ctz finds the next one
                        inv &= (inv - 1); // Standard trick to clear lowest set bit
                    }

                    // If inv is 0, we either finished the batch or ran out of terminators in this word.
                    // If we stopped because inv==0, we need to account for the trailing 1s in this word.
                    if (inv == 0) {
                        size_t word_end_bit = (word_idx + 1) * 64;
                        size_t valid_end = std::min(word_end_bit, limit);
                        
                        if (local_pos < valid_end) {
                            size_t remainder = valid_end - local_pos;
                            local_pending += remainder;
                            local_pos += remainder;
                        }

                        // Load next word
                        word_idx++;
                        bit_off = 0;
                        if (local_pos < limit) curr_word = ptr[word_idx];
                    }
                }
            }

            // Write back state
            pos_ = local_pos;
            pending_gap_ = local_pending;

            return produced;
        }

        void reset(size_t start_pos = 0) {
            pos_ = std::min(start_pos, total_bits_);
            pending_gap_ = 0;
        }
    };
} // namespace gef

#endif // GEF_FAST_UNARY_DECODER_HPP
