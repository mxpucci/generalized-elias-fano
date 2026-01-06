//
// Fast unary-decoder utility to reconstruct run-length encoded gaps.
// Refactored for High-Performance Hot Path.
// OPTIMIZED: 
// 1. Broadword (SWAR) for Reverse Order
// 2. 8-Bit Lookup Table (LUT) for Standard Order
//

#ifndef GEF_FAST_UNARY_DECODER_HPP
#define GEF_FAST_UNARY_DECODER_HPP

#include <algorithm>
#include <array>
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

    // Internal LUT Logic (Anonymous namespace to keep translation unit local/internal)
    namespace {
        struct LutEntry {
            uint8_t count;          // Number of SUBSEQUENT gaps in 'gaps' array (excluding first_gap)
            uint8_t remainder;      // Trailing 1s (new pending gap)
            uint8_t first_gap;      // Value of the first run of 1s (to fuse with pending)
            uint8_t gaps[7];        // Subsequent gaps found in this byte
        };

        constexpr LutEntry generate_lut_entry(uint8_t byte) {
            LutEntry entry{};
            int run = 0;
            bool first_found = false;
            
            // Scan bits LSB -> MSB (Standard Order)
            for (int i = 0; i < 8; ++i) {
                if ((byte >> i) & 1) { 
                    run++; // It's a 1 (gap part)
                } else {
                    // It's a 0 (terminator)
                    if (!first_found) {
                        entry.first_gap = run;
                        first_found = true;
                    } else {
                        entry.gaps[entry.count] = run;
                        entry.count++;
                    }
                    run = 0;
                }
            }
            // Whatever is left is the remainder (carry-over)
            entry.remainder = run;
            return entry;
        }

        // 256-entry table, fits easily in L1 cache (approx 2.5KB)
        constexpr std::array<LutEntry, 256> UNARY_LUT = [] {
            std::array<LutEntry, 256> table{};
            for (size_t i = 0; i < 256; ++i) table[i] = generate_lut_entry(static_cast<uint8_t>(i));
            return table;
        }();
    }

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

        // Single decode remains available
        inline uint64_t next() {
            uint32_t tmp;
            if (next_batch(&tmp, 1) == 1) return tmp;
            return 0;
        }

        inline void prefetch(size_t cache_lines_ahead = 4) const {
            #if defined(__GNUC__) || defined(__clang__)
                const size_t current_word_idx = pos_ >> 6;
                const size_t lookahead_word_idx = current_word_idx + (cache_lines_ahead * 8);
                if (lookahead_word_idx * 64 < total_bits_) {
                    __builtin_prefetch(data_ + lookahead_word_idx, 0, 1);
                }
            #endif
            }

        /**
         * @brief Decodes a batch of gaps.
         * ReverseOrder: Uses Broadword (SWAR) optimization.
         * StandardOrder: Uses 8-Bit Lookup Table (LUT) optimization.
         */
        size_t next_batch(uint32_t* out, size_t max_batch) {
            size_t produced = 0;
            const size_t limit = total_bits_;

            if (pos_ >= limit) return 0;

            if constexpr (ReverseOrder) {
                // --- REVERSE ORDER (MSB -> LSB) ---
                // Broadword Implementation
                
                size_t local_pos = pos_;
                size_t local_pending = pending_gap_;
            const uint64_t* ptr = data_;

            size_t word_idx = local_pos >> 6;
            uint32_t bit_off = static_cast<uint32_t>(local_pos & 63);
            uint64_t curr_word = ptr[word_idx];

            while (produced < max_batch && local_pos < limit) {
                size_t bits_available_in_word = 64 - bit_off;
                if (local_pos + bits_available_in_word > limit) {
                    bits_available_in_word = limit - local_pos;
                }

                    uint64_t inv = ~curr_word;
                    if (bit_off > 0) inv &= (~0ULL >> bit_off);
                    
                    size_t invalid_tail = 64 - (bit_off + bits_available_in_word);
                    if (invalid_tail > 0) inv &= (~0ULL << invalid_tail);

                    while (produced < max_batch && inv != 0) {
                        uint32_t lz = fast_clz(inv);
                        uint32_t run = lz - bit_off;
                        
                        out[produced++] = static_cast<uint32_t>(local_pending + run);
                        local_pending = 0;

                        uint32_t consumed = run + 1;
                        local_pos += consumed;
                        bit_off += consumed;
                        
                        inv &= ~(1ULL << (63 - lz));
                    }
                    
                    if (inv == 0) {
                        size_t bits_left_in_word = (word_idx + 1) * 64 - local_pos;
                        if (local_pos >= limit) bits_left_in_word = 0; 
                        else if (local_pos + bits_left_in_word > limit) bits_left_in_word = limit - local_pos;

                        if (bits_left_in_word > 0) {
                            local_pending += bits_left_in_word;
                            local_pos += bits_left_in_word;
                        }
                        
                        word_idx++;
                        bit_off = 0;
                        if (local_pos < limit) curr_word = ptr[word_idx];
                    }
                }
                pos_ = local_pos;
                pending_gap_ = local_pending;
                return produced;

                } else {
                // --- STANDARD ORDER (LSB -> MSB) ---
                // 8-Bit Lookup Table (LUT) Implementation
                
                uint32_t local_pending = static_cast<uint32_t>(pending_gap_);
                
                const uint8_t* base_ptr = reinterpret_cast<const uint8_t*>(data_);
                const uint8_t* current_ptr = base_ptr + (pos_ >> 3);
                
                if (total_bits_ == 0) return 0;
                const uint8_t* last_byte_ptr = base_ptr + ((total_bits_ - 1) >> 3);
                
                uint8_t bit_off = pos_ & 7;

                // 1. Handle Unaligned Start
                if (bit_off != 0) {
                     if (current_ptr > last_byte_ptr) return 0;

                    uint8_t byte = *current_ptr;
                    byte >>= bit_off;
                    if (bit_off > 0) byte |= (0xFF << (8 - bit_off)); 
                    
                    if (current_ptr == last_byte_ptr) {
                        size_t valid_bits_total = total_bits_ & 7;
                        if (valid_bits_total == 0) valid_bits_total = 8;
                        
                        if (valid_bits_total > bit_off) {
                             size_t valid_len = valid_bits_total - bit_off;
                             if (valid_len < 8) byte |= (~0u << valid_len);
                        } else {
                             byte = 0xFF;
                        }
                    }

                    if (byte != 0xFF) {
                        const auto& entry = UNARY_LUT[byte];
                        
                        int needed = 1 + entry.count;
                        if (produced + needed <= max_batch) {
                            out[produced++] = static_cast<uint32_t>(local_pending + entry.first_gap);
                            local_pending = 0;
                            for (int i = 0; i < entry.count; ++i) out[produced++] = entry.gaps[i];
                            
                            if (current_ptr == last_byte_ptr) {
                                 size_t valid_bits_total = total_bits_ & 7;
                                 if (valid_bits_total == 0) valid_bits_total = 8;
                                 if (valid_bits_total > bit_off) {
                                     size_t valid_len = valid_bits_total - bit_off;
                                     local_pending = entry.remainder - (8 - valid_len);
                                 } else {
                                     local_pending = 0;
                                 }
                            } else {
                                 local_pending = entry.remainder - bit_off;
                            }
                            current_ptr++;
                        } else {
                             out[produced++] = static_cast<uint32_t>(local_pending + entry.first_gap);
                        local_pending = 0;

                             if (produced == max_batch) {
                                 pos_ += (entry.first_gap + 1);
                                 pending_gap_ = 0;
                                 return produced;
                             }
                             
                             uint32_t bits_consumed = entry.first_gap + 1;
                             for (int i = 0; i < entry.count; ++i) {
                                 out[produced++] = entry.gaps[i];
                                 bits_consumed += entry.gaps[i] + 1;
                                 if (produced == max_batch) {
                                     pos_ += bits_consumed;
                                     pending_gap_ = 0;
                                     return produced;
                                 }
                             }
                             return produced;
                        }
                    } else {
                         if (current_ptr == last_byte_ptr) {
                             size_t valid_bits_total = total_bits_ & 7;
                             if (valid_bits_total == 0) valid_bits_total = 8;
                             if (valid_bits_total > bit_off) {
                                 local_pending += (valid_bits_total - bit_off);
                             }
                         } else {
                             local_pending += (8 - bit_off);
                         }
                         current_ptr++;
                    }
                }

                // 2. Hot Loop (Full Bytes)
                while (produced < max_batch && current_ptr < last_byte_ptr) {
                    uint8_t byte = *current_ptr;
                    
                    if (byte == 0xFF) {
                        local_pending += 8;
                        current_ptr++;
                        continue;
                    }

                    const auto& entry = UNARY_LUT[byte];
                    int needed = 1 + entry.count;

                    if (produced + needed <= max_batch) {
                        out[produced++] = static_cast<uint32_t>(local_pending + entry.first_gap);
                        local_pending = 0;
                        for (int i = 0; i < entry.count; ++i) out[produced++] = entry.gaps[i];
                        local_pending += entry.remainder;
                        current_ptr++;
                    } else {
                        out[produced++] = static_cast<uint32_t>(local_pending + entry.first_gap);
                        local_pending = 0;
                        uint32_t bits_consumed = entry.first_gap + 1;

                        if (produced == max_batch) {
                            pos_ = (current_ptr - base_ptr) * 8 + bits_consumed;
                            pending_gap_ = 0;
                            return produced;
                        }

                        for (int i = 0; i < entry.count; ++i) {
                            out[produced++] = entry.gaps[i];
                            bits_consumed += entry.gaps[i] + 1;
                            if (produced == max_batch) {
                                pos_ = (current_ptr - base_ptr) * 8 + bits_consumed;
                                pending_gap_ = 0;
                                return produced;
                            }
                        }
                    }
                }
                
                // 3. Last Byte
                if (produced < max_batch && current_ptr == last_byte_ptr) {
                     uint8_t byte = *current_ptr;
                     size_t valid_bits = total_bits_ & 7;
                     if (valid_bits == 0) valid_bits = 8;
                     
                     if (valid_bits < 8) byte |= (~0u << valid_bits);
                     
                     if (byte != 0xFF) {
                         const auto& entry = UNARY_LUT[byte];
                         int needed = 1 + entry.count;
                         
                         if (produced + needed <= max_batch) {
                             out[produced++] = static_cast<uint32_t>(local_pending + entry.first_gap);
                             local_pending = 0;
                             for (int i = 0; i < entry.count; ++i) out[produced++] = entry.gaps[i];
                             local_pending += (entry.remainder - (8 - valid_bits));
                             current_ptr++;
                         } else {
                             out[produced++] = static_cast<uint32_t>(local_pending + entry.first_gap);
                             local_pending = 0;
                             uint32_t bits_consumed = entry.first_gap + 1;
                             
                             if (produced == max_batch) {
                                 pos_ = (current_ptr - base_ptr) * 8 + bits_consumed;
                                 pending_gap_ = 0;
                                 return produced;
                             }
                             
                             for (int i = 0; i < entry.count; ++i) {
                                 out[produced++] = entry.gaps[i];
                                 bits_consumed += entry.gaps[i] + 1;
                                 if (produced == max_batch) {
                                     pos_ = (current_ptr - base_ptr) * 8 + bits_consumed;
                                     pending_gap_ = 0;
                                     return produced;
                                 }
                             }
                         }
                     } else {
                         local_pending += valid_bits;
                         current_ptr++;
                     }
                }

                // Writeback State
                pos_ = (current_ptr - base_ptr) * 8;
            pending_gap_ = local_pending;

            return produced;
            }
        }

        void reset(size_t start_pos = 0) {
            pos_ = std::min(start_pos, total_bits_);
            pending_gap_ = 0;
        }
    };
} // namespace gef

#endif // GEF_FAST_UNARY_DECODER_HPP
