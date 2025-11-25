//
// Fast unary-decoder utility to reconstruct run-length encoded gaps.
//

#ifndef GEF_FAST_UNARY_DECODER_HPP
#define GEF_FAST_UNARY_DECODER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits> // For std::is_same_v logic if needed elsewhere

#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
#include <immintrin.h>
#elif defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
#include <arm_neon.h>
#endif

namespace gef {
    
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

        // Standard: Mask keeps lowest 'bits' (Clear top)
        static inline uint64_t mask_low_bits(size_t bits) {
            return (~0ULL >> (64 - bits));
        }

        // Reverse: Mask keeps highest 'bits' (Clear bottom)
        static inline uint64_t mask_high_bits(size_t bits) {
            return (~0ULL << (64 - bits));
        }

    public:
        FastUnaryDecoder(const uint64_t* data, size_t total_bits, size_t start_pos = 0)
            : data_(data),
              total_bits_(total_bits),
              pos_(std::min(start_pos, total_bits)) {}

        inline uint64_t next() {
            if (!data_ || total_bits_ == 0) return 0;

            uint64_t count = 0;
            while (pos_ < total_bits_) {
                const size_t word_idx = pos_ >> 6;
                const uint32_t bit_offset = static_cast<uint32_t>(pos_ & 63);
                const size_t bits_rem = std::min<size_t>(64 - bit_offset, total_bits_ - pos_);
                
                uint64_t word = data_[word_idx];
                uint64_t inverted; 

                if constexpr (ReverseOrder) {
                    // Pasta Mode (MSB -> LSB)
                    // We align the current bit of interest to the MSB (Shift Left).
                    // Then we count leading zeros.
                    uint64_t chunk = word << bit_offset;
                    
                    // If we are at the end of the stream, we must ignore garbage bits at the bottom
                    if (bits_rem < 64) {
                        chunk &= mask_high_bits(bits_rem);
                    }
                    
                    inverted = ~chunk;
                    
                    // In Reverse mode, we might have masked out the bottom with 0s.
                    // Inverting turns them to 1s. But we are looking for 0s in original (1s in inverted).
                    // The valid area is the top 'bits_rem'.
                    // If inverted has a 0 in the valid area, clz will find it.
                    // If the valid area is all 1s (original was 0s), clz will be 0.
                    // Wait, Unary code: 11110.
                    // We look for 0.
                    // ~chunk: 00001....
                    // clz(~chunk) = 4. Correct.
                    
                    // Special case: if bits_rem < 64, the bottom bits of chunk were masked to 0.
                    // ~chunk makes them 1. clz stops at the first 1.
                    // If the valid part was all 0s (original 1s), we want to proceed.
                    // clz works fine because the padding (1s) won't trigger "leading zeros".
                    // However, we must ensure we don't read past bits_rem.
                } else {
                    // Standard Mode (LSB -> MSB)
                    // We align current bit to LSB (Shift Right).
                    uint64_t chunk = word >> bit_offset;
                    if (bits_rem < 64) {
                        chunk &= mask_low_bits(bits_rem);
                    }
                    inverted = ~chunk;
                    // Standard mask logic requires ensuring upper bits don't trigger ctz.
                    // Inverted upper bits will be 1s (due to mask clearing chunk to 0).
                    // ctz stops at first 1. Perfect.
                }

                if constexpr (ReverseOrder) {
                    // Look for 0 in original -> 1 in inverted
                    // But we want to count how many 1s were in original (gap size).
                    // Original: 1110...
                    // Inverted: 0001...
                    // clz returns 3. This is the gap size.
                    
                    // If the word was ALL 1s (11111...), Inverted is 00000...
                    // clz returns 64.
                    
                    // We need to check if we found the terminator (0) within bits_rem.
                    // Since we shifted left, we operate on the full 64-bit register.
                    
                    if (inverted != 0ULL) {
                        uint64_t run = __builtin_clzll(inverted);
                        if (run < bits_rem) {
                            pos_ += run + 1;
                            count += run;
                            return count;
                        }
                    }
                } else {
                    if (inverted != 0ULL) {
                        uint64_t run = __builtin_ctzll(inverted);
                        if (run < bits_rem) {
                            pos_ += run + 1;
                            count += run;
                            return count;
                        }
                    }
                }

                pos_ += bits_rem;
                count += bits_rem;
            }
            return count;
        }

        size_t next_batch(uint32_t* out, size_t max_batch) {
            if (!out || max_batch == 0 || !data_ || total_bits_ == 0) return 0;

#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
            constexpr size_t lane_cap = 8;
#elif defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
            constexpr size_t lane_cap = 4;
#else
            constexpr size_t lane_cap = 1;
#endif

            uint32_t lane_buffer[lane_cap];
            size_t lane_fill = 0;
            size_t produced = 0;
            size_t pending_gap = 0;

            // Helper to flush buffer to output
            auto flush_lane = [&]() {
                if (lane_fill == 0) return;
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
                if (lane_fill == lane_cap && produced + lane_cap <= max_batch) {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + produced), 
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_buffer)));
                    produced += lane_cap;
                    lane_fill = 0;
                    return;
                }
#elif defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
                if (lane_fill == lane_cap && produced + lane_cap <= max_batch) {
                    vst1q_u32(reinterpret_cast<uint32_t*>(out + produced), vld1q_u32(lane_buffer));
                    produced += lane_cap;
                    lane_fill = 0;
                    return;
                }
#endif
                for (size_t i = 0; i < lane_fill && produced < max_batch; ++i) {
                    out[produced++] = lane_buffer[i];
                }
                lane_fill = 0;
            };

            // Helper to push a gap into buffer
            auto emit_gap = [&](uint32_t gap) {
                lane_buffer[lane_fill++] = gap;
                if (lane_fill == lane_cap || (produced + lane_fill) == max_batch) {
                    flush_lane();
                }
            };

            while ((produced + lane_fill) < max_batch && pos_ < total_bits_) {
                const size_t word_idx = pos_ >> 6;
                const uint32_t bit_offset = static_cast<uint32_t>(pos_ & 63);
                size_t bits_rem = std::min<size_t>(64 - bit_offset, total_bits_ - pos_);
                
                uint64_t chunk;
                
                if constexpr (ReverseOrder) {
                    // --- REVERSE (Pasta) LOGIC ---
                    // Align to MSB
                    chunk = data_[word_idx] << bit_offset;
                    // Mask bottom garbage if near end
                    if (bits_rem < 64) chunk &= mask_high_bits(bits_rem);

                    while (bits_rem > 0) {
                        // Looking for 0 in chunk (which represents 1s in unary)
                        // Invert so we look for 1
                        uint64_t inv = ~chunk;
                        
                        // If bits_rem < 64, the bottom of 'chunk' is 0s. 
                        // So bottom of 'inv' is 1s.
                        // We must ignore these false positives.
                        // However, clz searches from TOP. Bottom 1s don't matter unless top is all 0s.
                        
                        if (inv == 0ULL) {
                            // All 1s in the valid part (and implicit 1s in the padding)
                            // But wait, if bits_rem < 64, chunk had 0s at bottom. ~chunk has 1s at bottom.
                            // inv can only be 0ULL if chunk was ALL 1s (0xFFFFFF...)
                            // This means current word is all continuation.
                            pending_gap += bits_rem;
                            pos_ += bits_rem;
                            bits_rem = 0;
                            break;
                        }

                        uint64_t run = __builtin_clzll(inv);
                        
                        if (run >= bits_rem) {
                            // No terminator found in the VALID part of the word
                            pending_gap += bits_rem;
                            pos_ += bits_rem;
                            bits_rem = 0;
                            break;
                        }

                        // Terminator found
                        pending_gap += run;
                        emit_gap(static_cast<uint32_t>(pending_gap));
                        pending_gap = 0;

                        // Advance
                        // We consumed 'run' ones and 1 terminator zero.
                        size_t consumed = run + 1;
                        pos_ += consumed;
                        bits_rem -= consumed;
                        
                        // Shift out processed bits (Shift Left in Reverse mode)
                        chunk <<= consumed;
                        
                        // Check if batch is full
                        if ((produced + lane_fill) >= max_batch) {
                            flush_lane();
                            return produced;
                        }
                    }

                } else {
                    // --- STANDARD (SDSL) LOGIC ---
                    // Align to LSB
                    chunk = data_[word_idx] >> bit_offset;
                    if (bits_rem < 64) chunk &= mask_low_bits(bits_rem);

                    while (bits_rem > 0) {
                        uint64_t inv = ~chunk;
                        
                        // Mask high garbage to avoid false ctz hits
                        // If bits_rem < 64, high bits of chunk are 0.
                        // inv high bits are 1.
                        // ctz works from bottom, so high 1s don't matter unless bottom is all 0s.
                        
                        // If chunk was all 1s (valid part), inv is all 0s (valid part) + 1s (garbage).
                        // To detect "All 1s in valid part", we need to check if ctz >= bits_rem.
                        
                        if ((inv & mask_low_bits(bits_rem)) == 0ULL) {
                             // Valid part is all 1s (no terminator)
                             pending_gap += bits_rem;
                             pos_ += bits_rem;
                             bits_rem = 0;
                             break;
                        }

                        uint64_t run = __builtin_ctzll(inv);

                        if (run >= bits_rem) {
                            pending_gap += bits_rem;
                            pos_ += bits_rem;
                            bits_rem = 0;
                            break;
                        }

                        // Terminator found
                        pending_gap += run;
                        emit_gap(static_cast<uint32_t>(pending_gap));
                        pending_gap = 0;

                        size_t consumed = run + 1;
                        pos_ += consumed;
                        bits_rem -= consumed;
                        
                        // Shift out processed bits (Shift Right in Standard mode)
                        chunk >>= consumed;

                        if ((produced + lane_fill) >= max_batch) {
                            flush_lane();
                            return produced;
                        }
                    }
                }
            }

            flush_lane();
            return produced;
        }

        void reset(size_t start_pos = 0) {
            pos_ = std::min(start_pos, total_bits_);
        }
    };
} // namespace gef

#endif // GEF_FAST_UNARY_DECODER_HPP