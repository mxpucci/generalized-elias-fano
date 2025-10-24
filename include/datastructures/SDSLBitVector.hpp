#ifndef SDSLBITVECTOR_HPP
#define SDSLBITVECTOR_HPP

#include "IBitVector.hpp"
#include <sdsl/bit_vectors.hpp>
#include <sdsl/rank_support.hpp>
#include <sdsl/select_support.hpp>
#include <memory>
#include <fstream>
#include <vector>
#include <algorithm> // For std::fill

// Include headers for SIMD intrinsics
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/**
 * @brief SDSL-based implementation of IBitVector
 * 
 * Wraps sdsl::bit_vector with optional rank/select support structures.
 * Rank and select operations are only available if explicitly enabled.
 */
class SDSLBitVector : public IBitVector {
private:
    sdsl::bit_vector bv_;
    
    std::unique_ptr<sdsl::rank_support_v5<1>> rank_support_;
    std::unique_ptr<sdsl::select_support_mcl<1>> select1_support_;
    std::unique_ptr<sdsl::select_support_mcl<0>> select0_support_;
    

public:
    static double rank_overhead_per_bit() { return 0.0625; }
    static double select1_overhead_per_bit() { return 0.2; }
    static double select0_overhead_per_bit() { return 0.2; }

    // Bring inherited methods into scope to avoid name hiding
    using IBitVector::rank;
    using IBitVector::rank0;
    using IBitVector::serialize;

    SDSLBitVector(sdsl::bit_vector bv) 
        : IBitVector(), bv_(std::move(bv)) {
    }

    SDSLBitVector(size_t size)
        : SDSLBitVector(sdsl::bit_vector(size, 0)) {}

    SDSLBitVector(const std::vector<bool>& bits) : IBitVector() {
        bv_ = sdsl::bit_vector(bits.size(), 0);
        uint64_t* data = bv_.data();
        size_t n = bits.size();
        size_t full_words = n >> 6;
        size_t i = 0;

        for (size_t w = 0; w < full_words; ++w) {
            uint64_t x = 0;
            for (uint32_t b = 0; b < 64; ++b, ++i) {
                x |= static_cast<uint64_t>(bits[i]) << b;
            }
            data[w] = x;
        }

        uint32_t rem = static_cast<uint32_t>(n & 63);
        if (rem) {
            uint64_t x = 0;
            for (uint32_t b = 0; b < rem; ++b, ++i) {
                x |= static_cast<uint64_t>(bits[i]) << b;
            }
            data[full_words] = x;
        }
    }

    // Rule of 5: Destructor
    ~SDSLBitVector() = default;

    // Rule of 5: Copy constructor
    SDSLBitVector(const SDSLBitVector& other) : IBitVector(), bv_(other.bv_) {
        if (other.rank_support_) {
            rank_support_ = std::make_unique<sdsl::rank_support_v5<1>>(&bv_);
        }
        if (other.select1_support_) {
            select1_support_ = std::make_unique<sdsl::select_support_mcl<1>>(&bv_);
        }
        if (other.select0_support_) {
            select0_support_ = std::make_unique<sdsl::select_support_mcl<0>>(&bv_);
        }
    }

    // Rule of 5: Copy assignment operator
    SDSLBitVector& operator=(const SDSLBitVector& other) {
        if (this != &other) {
            bv_ = other.bv_;
            rank_support_.reset();
            select1_support_.reset();
            select0_support_.reset();
            
            if (other.rank_support_) {
                rank_support_ = std::make_unique<sdsl::rank_support_v5<1>>(&bv_);
            }
            if (other.select1_support_) {
                select1_support_ = std::make_unique<sdsl::select_support_mcl<1>>(&bv_);
            }
            if (other.select0_support_) {
                select0_support_ = std::make_unique<sdsl::select_support_mcl<0>>(&bv_);
            }
        }
        return *this;
    }

    // Rule of 5: Move constructor
    SDSLBitVector(SDSLBitVector&& other) noexcept : IBitVector(), bv_(std::move(other.bv_)) {
        if (other.rank_support_) {
            rank_support_ = std::make_unique<sdsl::rank_support_v5<1>>(&bv_);
        }
        if (other.select1_support_) {
            select1_support_ = std::make_unique<sdsl::select_support_mcl<1>>(&bv_);
        }
        if (other.select0_support_) {
            select0_support_ = std::make_unique<sdsl::select_support_mcl<0>>(&bv_);
        }
    }

    // Rule of 5: Move assignment operator
    SDSLBitVector& operator=(SDSLBitVector&& other) noexcept {
        if (this != &other) {
            bv_ = std::move(other.bv_);
            rank_support_.reset();
            select1_support_.reset();
            select0_support_.reset();
            
            if (other.rank_support_) {
                rank_support_ = std::make_unique<sdsl::rank_support_v5<1>>(&bv_);
            }
            if (other.select1_support_) {
                select1_support_ = std::make_unique<sdsl::select_support_mcl<1>>(&bv_);
            }
            if (other.select0_support_) {
                select0_support_ = std::make_unique<sdsl::select_support_mcl<0>>(&bv_);
            }
        }
        return *this;
    }

    bool operator[](size_t index) const override {
        return bv_[index];
    }

    void set(size_t index, bool value) override {
        bv_[index] = value;
    }

    void set_range(size_t start, size_t count, bool value) override {
        if (count == 0) return;
        if (start + count > size()) {
            throw std::out_of_range("set_range writes out of bounds");
        }

        uint64_t* data = bv_.data();
        const size_t end = start + count - 1;

        const size_t start_word = start >> 6;
        const size_t end_word   = end   >> 6;
        const uint32_t start_off = static_cast<uint32_t>(start & 63);
        const uint32_t end_off   = static_cast<uint32_t>(end   & 63);

        const uint64_t fill_val = value ? ~0ULL : 0ULL;

        if (start_word == end_word) {
            // Single word logic remains the same
            const uint64_t mask = (count == 64) ? ~0ULL : ((1ULL << count) - 1ULL) << start_off;
            if (value) data[start_word] |= mask;
            else       data[start_word] &= ~mask;
            return;
        }

        // Head: Partial first word
        const uint64_t head_mask = ~0ULL << start_off;
        if (value) data[start_word] |= head_mask;
        else       data[start_word] &= ~head_mask;

        // Body: Full words, optimized with SIMD
        if (end_word > start_word + 1) {
            uint64_t* body_begin = data + start_word + 1;
            const size_t body_words = end_word - (start_word + 1);

#if defined(__AVX2__)
            __m256i val_vec = _mm256_set1_epi64x(fill_val);
            size_t i = 0;
            // Process 4 words (32 bytes) at a time
            for (; i + 4 <= body_words; i += 4) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(body_begin + i), val_vec);
            }
            // Handle remaining 0-3 words
            for (; i < body_words; ++i) {
                body_begin[i] = fill_val;
            }
#elif defined(__ARM_NEON)
            uint64x2_t val_vec = vdupq_n_u64(fill_val);
            size_t i = 0;
            // Process 2 words (16 bytes) at a time
            for (; i + 2 <= body_words; i += 2) {
                vst1q_u64(body_begin + i, val_vec);
            }
            // Handle remaining 0-1 word
            for (; i < body_words; ++i) {
                body_begin[i] = fill_val;
            }
#else
            // Fallback for other architectures or if SIMD is disabled
            std::fill(body_begin, body_begin + body_words, fill_val);
#endif
        }

        // Tail: Partial last word
        const uint64_t tail_mask = (end_off == 63) ? ~0ULL : ((1ULL << (end_off + 1)) - 1ULL);
        if (value) data[end_word] |= tail_mask;
        else       data[end_word] &= ~tail_mask;
    }

    inline void set_bits(size_t start_index, uint64_t bits, uint8_t num_bits) override {
        if (num_bits == 0) return;
        if (num_bits > 64) throw std::invalid_argument("num_bits cannot exceed 64");
        if (start_index + num_bits > size()) throw std::out_of_range("set_bits out of bounds");

        uint64_t* data = bv_.data();
        const size_t w = start_index >> 6;
        const uint32_t off = static_cast<uint32_t>(start_index & 63);

        if (off + num_bits <= 64) {
            uint64_t mask = (num_bits == 64) ? ~0ULL : ((1ULL << num_bits) - 1ULL);
            mask <<= off;
            uint64_t v = (bits << off) & mask;
            data[w] = (data[w] & ~mask) | v;
        } else {
            // Split across two words
            const uint32_t left = 64 - off;
            const uint32_t right = num_bits - left;

            // word w
            uint64_t mask0 = ~0ULL << off;
            uint64_t v0 = (bits << off) & mask0;
            data[w] = (data[w] & ~mask0) | v0;

            // word w+1
            uint64_t mask1 = (right == 64) ? ~0ULL : ((1ULL << right) - 1ULL);
            uint64_t v1 = (bits >> left) & mask1;
            data[w+1] = (data[w+1] & ~mask1) | v1;
        }
    }

    /**
     * @brief Set/get bit at given index (non-const version)
     */
    auto operator[](size_t index) -> decltype(bv_[index]) {
        return bv_[index];
    }

    size_t size() const override {
        return bv_.size();
    }

    size_t rank(size_t pos) const override {
        if (!rank_support_) {
            throw std::runtime_error("Rank support not enabled");
        }
        return (*rank_support_)(pos);
    }

    size_t select(size_t k) const override {
        if (!select1_support_) {
            throw std::runtime_error("Select support not enabled");
        }
        return (*select1_support_)(k);
    }

    size_t select0(size_t k) const override {
        if (!select0_support_) {
            throw std::runtime_error("Select support not enabled");
        }
        return (*select0_support_)(k);
    }

    size_t size_in_bytes() const override {
        size_t total = sdsl::size_in_bytes(bv_);
        if (rank_support_) {
            total += sdsl::size_in_bytes(*rank_support_);
        }
        if (select1_support_) {
            total += sdsl::size_in_bytes(*select1_support_);
        }
        if (select0_support_) {
            total += sdsl::size_in_bytes(*select0_support_);
        }
        return total;
    }

    size_t support_size_in_bytes() const override {
        size_t total = 0;
        if (rank_support_) {
            total += sdsl::size_in_bytes(*rank_support_);
        }
        if (select1_support_) {
            total += sdsl::size_in_bytes(*select1_support_);
        }
        if (select0_support_) {
            total += sdsl::size_in_bytes(*select0_support_);
        }
        return total;
    }

    size_t size_in_megabytes() const override {
        return (size_in_bytes() + 1024 * 1024 - 1) / (1024 * 1024);
    }

    void serialize(std::ofstream& out) const override {
        bv_.serialize(out);
    }


    static SDSLBitVector load(std::ifstream& in) {
        if (!in) {
            throw std::runtime_error("Cannot open stream");
        }

        sdsl::bit_vector bv;
        bv.load(in);

        SDSLBitVector result(std::move(bv));
        return result;
    }

    static SDSLBitVector load(const std::filesystem::path& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        return load(in);
    }

    std::unique_ptr<IBitVector> clone() const override {
        return std::make_unique<SDSLBitVector>(*this);
    }

    void enable_rank() override {
        if (!rank_support_) {
            rank_support_ = std::make_unique<sdsl::rank_support_v5<1>>(&bv_);
        }
    }
    void enable_select1() override {
        if (!select1_support_) {
            select1_support_ = std::make_unique<sdsl::select_support_mcl<1>>(&bv_);
        }
    }

    void enable_select0() override {
        if (!select0_support_) {
            select0_support_ = std::make_unique<sdsl::select_support_mcl<0>>(&bv_);
        }
    }

};

#endif