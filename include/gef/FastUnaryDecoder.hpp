//
// Fast unary-decoder utility to reconstruct run-length encoded gaps.
// Refactored for High-Performance Hot Path.
//

#ifndef GEF_FAST_UNARY_DECODER_HPP
#define GEF_FAST_UNARY_DECODER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// Includes for SIMD/Bit manipulation intrinsics
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace gef {

    /**
     * @brief Helper to count trailing zeros (CTZ).
     * Used for Standard Mode (LSB -> MSB).
     * Note: Input 0 is handled by caller (usually), but if passed, behavior depends on intrinsic.
     * We return 64 for 0 to be safe.
     */
    static inline uint64_t count_trailing_zeros(uint64_t x) {
        if (x == 0) return 64;
#if defined(__BMI__)
        return _tzcnt_u64(x);
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_ctzll(x);
#elif defined(_MSC_VER) && defined(_M_X64)
        return _tzcnt_u64(x);
#else
        unsigned long result;
        if (_BitScanForward64(&result, x)) return result;
        return 64;
#endif
    }

    /**
     * @brief Helper to count leading zeros (CLZ).
     * Used for Reverse Mode (MSB -> LSB).
     */
    static inline uint64_t count_leading_zeros(uint64_t x) {
        if (x == 0) return 64;
#if defined(__BMI__)
        return _lzcnt_u64(x);
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_clzll(x);
#elif defined(_MSC_VER) && defined(_M_X64)
        return _lzcnt_u64(x);
#else
        unsigned long result;
        if (_BitScanReverse64(&result, x)) return 63 - result;
        return 64;
#endif
    }

    /**
     * @brief Fast decoder for Unary codes (gaps).
     * @tparam ReverseOrder 
     * false (Default/SDSL): Bits read LSB -> MSB (Standard).
     * true (Pasta): Bits read MSB -> LSB (Reverse).
     */
    template<bool ReverseOrder = false>
    class FastUnaryDecoder {
    private:
        const uint64_t* data_;
        size_t total_bits_;
        size_t pos_;

    public:
        FastUnaryDecoder(const uint64_t* data, size_t total_bits, size_t start_pos = 0)
            : data_(data),
              total_bits_(total_bits),
              pos_(std::min(start_pos, total_bits)) {}

        /**
         * @brief Decodes a single gap.
         */
        inline uint64_t next() {
            if (pos_ >= total_bits_) return 0;

            uint64_t count = 0;
            while (pos_ < total_bits_) {
                const size_t word_idx = pos_ >> 6;
                const unsigned bit_off = pos_ & 63;
                unsigned valid_bits = 64 - bit_off;

                uint64_t word = data_[word_idx];
                uint64_t buffer = ~word; // Invert: Gap=0, Term=1

                // Align buffer
                if constexpr (ReverseOrder) {
                    buffer <<= bit_off;
                } else {
                    buffer >>= bit_off;
                }

                if (buffer != 0) {
                    uint64_t run;
                    if constexpr (ReverseOrder) {
                        run = count_leading_zeros(buffer);
                    } else {
                        run = count_trailing_zeros(buffer);
                    }

                    if (run < valid_bits) {
                        // Found terminator
                        // Check bounds for last word
                        if (pos_ + run >= total_bits_) {
                            // Terminator in garbage area
                            count += (total_bits_ - pos_);
                            pos_ = total_bits_;
                            return count;
                        }
                        
                        count += run;
                        pos_ += run + 1;
                        return count;
                    }
                }

                // Not found in valid part of this word
                size_t bits_rem = std::min<size_t>(valid_bits, total_bits_ - pos_);
                count += bits_rem;
                pos_ += bits_rem;
            }
            return count;
        }

        /**
         * @brief Decodes a batch of gaps directly into 'out'.
         * Optimized for hot path with localized state and minimal branching.
         */
        size_t next_batch(uint32_t* out, size_t max_batch) {
            if (!out || max_batch == 0 || pos_ >= total_bits_) return 0;

            size_t produced = 0;

            // 1. Initialize Local State
            size_t local_pos = pos_;
            
            // Current word pointer and end pointer
            const uint64_t* ptr = data_ + (local_pos >> 6);
            const uint64_t* end_ptr = data_ + ((total_bits_ - 1) >> 6);

            unsigned bit_off = local_pos & 63;
            unsigned valid_bits = 64 - bit_off;

            // 2. Pre-load buffer (Inverted)
            // Original: 1=gap, 0=term. Inverted: 0=gap, 1=term.
            uint64_t buffer = ~(*ptr);

            if constexpr (ReverseOrder) {
                buffer <<= bit_off; // MSB align
            } else {
                buffer >>= bit_off; // LSB align
            }

            uint32_t pending_gap = 0;

            // 3. Hot Loop
            while (produced < max_batch) {
                // If buffer is 0, it means all remaining valid bits are gaps (0s).
                // Or we shifted in 0s (gaps) and valid part was also 0s.
                if (buffer == 0) {
                    pending_gap += valid_bits;
                    local_pos += valid_bits;

                    // Move to next word
                    if (ptr >= end_ptr) {
                        break; // End of stream
                    }
                    ptr++;
                    // Reload full word
                    buffer = ~(*ptr);
                    valid_bits = 64; 
                } else {
                    uint64_t run;
                    if constexpr (ReverseOrder) {
                        run = count_leading_zeros(buffer);
                    } else {
                        run = count_trailing_zeros(buffer);
                    }

                    if (run < valid_bits) {
                        // FOUND TERMINATOR
                        // Check strict bounds if we are at the last word
                        if (ptr == end_ptr && (local_pos + run >= total_bits_)) {
                            // Terminator is outside valid stream (garbage)
                            local_pos = total_bits_;
                            break;
                        }

                        out[produced++] = pending_gap + run;
                        pending_gap = 0;

                        unsigned consumed = run + 1;
                        local_pos += consumed;
                        valid_bits -= consumed;

                        // Shift out consumed bits
                        // Handle UB if valid_bits becomes 0 (shifted by 64)
                        if (valid_bits == 0) {
                            buffer = 0; // Force refill next iter
                        } else {
                            if constexpr (ReverseOrder) {
                                buffer <<= consumed;
                            } else {
                                buffer >>= consumed;
                            }
                        }
                    } else {
                        // False positive: Terminator found is beyond valid_bits.
                        // This happens because we shifted in 0s (gaps) which don't trigger find,
                        // but if the valid part was all 0s, ctz/clz might return something >= valid_bits.
                        
                        pending_gap += valid_bits;
                        local_pos += valid_bits;

                        if (ptr >= end_ptr) {
                            break;
                        }
                        ptr++;
                        buffer = ~(*ptr);
                        valid_bits = 64;
                    }
                }
            }

            // Sync state back
            pos_ = local_pos;
            return produced;
        }

        void reset(size_t start_pos = 0) {
            pos_ = std::min(start_pos, total_bits_);
        }
    };
} // namespace gef

#endif // GEF_FAST_UNARY_DECODER_HPP
