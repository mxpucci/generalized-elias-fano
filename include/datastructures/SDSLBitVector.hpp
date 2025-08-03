#ifndef SDSLBITVECTOR_HPP
#define SDSLBITVECTOR_HPP

#include "IBitVector.hpp"
#include <sdsl/bit_vectors.hpp>
#include <sdsl/rank_support.hpp>
#include <sdsl/select_support.hpp>
#include <memory>
#include <fstream>
#include <vector>

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
        bv_ = sdsl::bit_vector(bits.size());
        for (size_t i = 0; i < bits.size(); ++i) {
            bv_[i] = bits[i];
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
        if (count == 0) {
            return;
        }
        if (start + count > size()) {
            throw std::out_of_range("set_range writes out of bounds");
        }

        uint64_t* data = bv_.data();
        size_t end = start + count - 1;

        const size_t start_word = start / 64;
        const size_t end_word = end / 64;
        const size_t start_offset = start % 64;
        const size_t end_offset = end % 64;

        if (start_word == end_word) {
            // Case 1: The entire range is within a single 64-bit word.
            // Create a mask for 'count' bits and shift it to the start_offset.
            uint64_t mask = (count == 64) ? ~0ULL : ((1ULL << count) - 1);
            mask <<= start_offset;

            if (value) {
                data[start_word] |= mask;
            } else {
                data[start_word] &= ~mask;
            }
        } else {
            // Case 2: The range spans multiple words.

            // Part 1: Handle the "head" in the first word.
            // Create a mask for bits from start_offset to 63.
            const uint64_t head_mask = ~0ULL << start_offset;
            if (value) {
                data[start_word] |= head_mask;
            } else {
                data[start_word] &= ~head_mask;
            }

            // Part 2: Handle the "body" of full words.
            const uint64_t fill_val = value ? ~0ULL : 0ULL;
            // Use std::fill for a fast loop over the full words.
            if (end_word > start_word + 1) {
                std::fill(data + start_word + 1, data + end_word, fill_val);
            }

            // Part 3: Handle the "tail" in the last word.
            // Create a mask for bits from 0 to end_offset.
            uint64_t tail_mask = sdsl::bits::lo_set[end_offset + 1];
            if (value) {
                data[end_word] |= tail_mask;
            } else {
                data[end_word] &= ~tail_mask;
            }
        }
    }

    void set_bits(size_t start_index, uint64_t bits, uint8_t num_bits) override {
        if (num_bits == 0) return;
        if (num_bits > 64) {
            throw std::invalid_argument("num_bits cannot exceed 64");
        }
        if (start_index + num_bits > size()) {
            throw std::out_of_range("set_bits out of bounds");
        }

        // Use SDSL's built-in integer setting method
        bv_.set_int(start_index, bits, num_bits);
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