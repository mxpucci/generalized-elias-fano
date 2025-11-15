//
// Fast unary-decoder utility to reconstruct run-length encoded gaps.
//

#ifndef GEF_FAST_UNARY_DECODER_HPP
#define GEF_FAST_UNARY_DECODER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
#include <immintrin.h>
#elif defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
#include <arm_neon.h>
#endif

namespace gef {
    class FastUnaryDecoder {
    private:
        const uint64_t* data_;
        size_t total_bits_;
        size_t pos_;

        static inline uint64_t mask_for_bits(size_t bits) {
            if (bits >= 64) {
                return std::numeric_limits<uint64_t>::max();
            }
            return (uint64_t(1) << bits) - 1;
        }

    public:
        FastUnaryDecoder(const uint64_t* data, size_t total_bits, size_t start_pos = 0)
            : data_(data),
              total_bits_(total_bits),
              pos_(std::min(start_pos, total_bits)) {}

        inline uint64_t next() {
            if (!data_ || total_bits_ == 0) {
                return 0;
            }

            uint64_t count = 0;
            while (pos_ < total_bits_) {
                const size_t word_idx = pos_ >> 6;
                const uint32_t bit_offset = static_cast<uint32_t>(pos_ & 63);
                const size_t bits_remaining_in_word = std::min<size_t>(64 - bit_offset, total_bits_ - pos_);
                const uint64_t word = data_[word_idx];
                const uint64_t chunk = (word >> bit_offset) & mask_for_bits(bits_remaining_in_word);
                const uint64_t inverted = (~chunk) & mask_for_bits(bits_remaining_in_word);

                if (inverted != 0ULL) {
                    const uint32_t zero_offset = static_cast<uint32_t>(__builtin_ctzll(inverted));
                    pos_ += zero_offset + 1; // consume trailing ones + terminator
                    count += zero_offset;
                    return count;
                }

                pos_ += bits_remaining_in_word;
                count += bits_remaining_in_word;
            }

            return count;
        }

        size_t next_batch(uint32_t* out, size_t max_batch) {
            if (!out || max_batch == 0 || !data_ || total_bits_ == 0) {
                return 0;
            }

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

            auto flush_lane = [&]() {
                if (lane_fill == 0) return;
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
                if (lane_fill == lane_cap && produced + lane_cap <= max_batch) {
                    __m256i vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lane_buffer));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + produced), vec);
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

            auto emit_gap = [&](uint32_t gap) {
                lane_buffer[lane_fill++] = gap;
                if (lane_fill == lane_cap || (produced + lane_fill) == max_batch) {
                    flush_lane();
                }
            };

            while ((produced + lane_fill) < max_batch && pos_ < total_bits_) {
                const size_t word_idx = pos_ >> 6;
                const uint32_t bit_offset = static_cast<uint32_t>(pos_ & 63);
                size_t bits_remaining_in_word = std::min<size_t>(64 - bit_offset, total_bits_ - pos_);
                uint64_t chunk = data_[word_idx] >> bit_offset;
                uint64_t chunk_mask = mask_for_bits(bits_remaining_in_word);
                chunk &= chunk_mask;

                while (bits_remaining_in_word > 0) {
                    const uint64_t zero_mask = (~chunk) & chunk_mask;
                    if (zero_mask == 0ULL) {
                        pending_gap += bits_remaining_in_word;
                        pos_ += bits_remaining_in_word;
                        bits_remaining_in_word = 0;
                        break;
                    }

                    const uint32_t zero_offset = static_cast<uint32_t>(__builtin_ctzll(zero_mask));
                    pending_gap += zero_offset;
                    pos_ += zero_offset + 1;
                    bits_remaining_in_word -= zero_offset + 1;
                    chunk >>= (zero_offset + 1);
                    chunk_mask = mask_for_bits(bits_remaining_in_word);
                    emit_gap(static_cast<uint32_t>(pending_gap));
                    pending_gap = 0;

                    if ((produced + lane_fill) >= max_batch) {
                        flush_lane();
                        return produced;
                    }

                    if (bits_remaining_in_word == 0) {
                        break;
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


