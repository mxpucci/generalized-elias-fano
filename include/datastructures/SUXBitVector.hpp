#ifndef SUXBITVECTOR_HPP
#define SUXBITVECTOR_HPP

#include "IBitVector.hpp"
#include <sux/bits/Rank9.hpp>
#include <sux/bits/SimpleSelectHalf.hpp>
#include <sux/bits/SimpleSelectZeroHalf.hpp>
#include <memory>
#include <optional>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>

/**
 * @brief SUX-based implementation of IBitVector
 * * OPTIMIZED: Uses std::optional for flat memory layout (no heap allocs).
 * * FIXED: Uses 'mutable' to handle SUX non-const methods within const interfaces.
 */
class SUXBitVector : public IBitVector {
private:
    std::vector<uint64_t> data_;
    size_t size_;
    
    // 'mutable' is required here because SUX methods (rank/select) are not marked const.
    // Unlike unique_ptr, std::optional propagates constness to the contained value.
    // 'mutable' allows us to call non-const SUX methods from const IBitVector methods.
    mutable std::optional<sux::bits::Rank9<>> rank_support_;
    mutable std::optional<sux::bits::SimpleSelectHalf<>> select1_support_;
    mutable std::optional<sux::bits::SimpleSelectZeroHalf<>> select0_support_;

public:
    static constexpr bool reverse_bit_order = false;
    static double rank_overhead_per_bit() { return 0.0625; }
    static double select1_overhead_per_bit() { return 0.1875; }
    static double select0_overhead_per_bit() { return 0.1875; }

    using IBitVector::rank;
    using IBitVector::rank0;
    using IBitVector::serialize;

    SUXBitVector(size_t size) : IBitVector(), size_(size) {
        size_t num_words = (size + 63) / 64;
        data_.resize(num_words, 0);
    }

    SUXBitVector(const std::vector<bool>& bits) : IBitVector(), size_(bits.size()) {
        size_t num_words = (size_ + 63) / 64;
        data_.resize(num_words, 0);
        
        for (size_t i = 0; i < size_; ++i) {
            if (bits[i]) {
                size_t word_idx = i / 64;
                size_t bit_idx = i % 64;
                data_[word_idx] |= (1ULL << bit_idx);
            }
        }
    }

    SUXBitVector(std::vector<uint64_t> data, size_t size) 
        : IBitVector(), data_(std::move(data)), size_(size) {
    }

    ~SUXBitVector() = default;

    // Copy constructor
    SUXBitVector(const SUXBitVector& other) 
        : IBitVector(), data_(other.data_), size_(other.size_) {
        // We must re-construct supports to point to OUR data_, not other.data_
        if (other.rank_support_) {
            rank_support_.emplace(data_.data(), size_);
        }
        if (other.select1_support_) {
            select1_support_.emplace(data_.data(), size_);
        }
        if (other.select0_support_) {
            select0_support_.emplace(data_.data(), size_);
        }
    }

    // Copy assignment operator
    SUXBitVector& operator=(const SUXBitVector& other) {
        if (this != &other) {
            data_ = other.data_;
            size_ = other.size_;
            
            rank_support_.reset();
            select1_support_.reset();
            select0_support_.reset();
            
            if (other.rank_support_) {
                rank_support_.emplace(data_.data(), size_);
            }
            if (other.select1_support_) {
                select1_support_.emplace(data_.data(), size_);
            }
            if (other.select0_support_) {
                select0_support_.emplace(data_.data(), size_);
            }
        }
        return *this;
    }

    // Rule of 5: Move constructor
    SUXBitVector(SUXBitVector&& other) noexcept : IBitVector() {
        // 1. Move the raw data
        // The underlying buffer address is preserved by std::vector move.
        data_ = std::move(other.data_);
        size_ = other.size_;
        
        // 2. Re-initialize supports via emplace (Fixes "deleted function" error)
        // We cannot simply assign the optional (rank_support_ = ...). 
        // We must construct the support in-place on the new object.
        if (other.rank_support_) {
            rank_support_.emplace(data_.data(), size_);
        }
        if (other.select1_support_) {
            select1_support_.emplace(data_.data(), size_);
        }
        if (other.select0_support_) {
            select0_support_.emplace(data_.data(), size_);
        }

        // 3. Reset other
        other.size_ = 0;
        other.rank_support_.reset();
        other.select1_support_.reset();
        other.select0_support_.reset();
    }

    // Rule of 5: Move assignment operator
    SUXBitVector& operator=(SUXBitVector&& other) noexcept {
        if (this != &other) {
            // 1. Move data
            data_ = std::move(other.data_);
            size_ = other.size_;
            
            // 2. Reset current supports
            rank_support_.reset();
            select1_support_.reset();
            select0_support_.reset();

            // 3. Re-initialize supports via emplace
            if (other.rank_support_) {
                rank_support_.emplace(data_.data(), size_);
            }
            if (other.select1_support_) {
                select1_support_.emplace(data_.data(), size_);
            }
            if (other.select0_support_) {
                select0_support_.emplace(data_.data(), size_);
            }

            // 4. Reset other
            other.size_ = 0;
            other.rank_support_.reset();
            other.select1_support_.reset();
            other.select0_support_.reset();
        }
        return *this;
    }

    bool operator[](size_t index) const override {
        if (index >= size_) {
            throw std::out_of_range("Index out of bounds");
        }
        size_t word_idx = index / 64;
        size_t bit_idx = index % 64;
        return (data_[word_idx] >> bit_idx) & 1ULL;
    }

    void set(size_t index, bool value) override {
        if (index >= size_) {
            throw std::out_of_range("Index out of bounds");
        }
        size_t word_idx = index / 64;
        size_t bit_idx = index % 64;
        if (value) {
            data_[word_idx] |= (1ULL << bit_idx);
        } else {
            data_[word_idx] &= ~(1ULL << bit_idx);
        }
    }

    void set_range(size_t start, size_t count, bool value) override {
        if (count == 0) return;
        if (start + count > size_) {
            throw std::out_of_range("set_range writes out of bounds");
        }

        const size_t end = start + count - 1;
        const size_t start_word = start / 64;
        const size_t end_word = end / 64;
        const uint32_t start_off = start % 64;
        const uint32_t end_off = end % 64;

        const uint64_t fill_val = value ? ~0ULL : 0ULL;

        if (start_word == end_word) {
            const uint64_t mask = (count == 64) ? ~0ULL : ((1ULL << count) - 1ULL) << start_off;
            if (value) {
                data_[start_word] |= mask;
            } else {
                data_[start_word] &= ~mask;
            }
            return;
        }

        const uint64_t head_mask = ~0ULL << start_off;
        if (value) {
            data_[start_word] |= head_mask;
        } else {
            data_[start_word] &= ~head_mask;
        }

        if (end_word > start_word + 1) {
            std::fill(data_.begin() + start_word + 1, data_.begin() + end_word, fill_val);
        }

        const uint64_t tail_mask = (end_off == 63) ? ~0ULL : ((1ULL << (end_off + 1)) - 1ULL);
        if (value) {
            data_[end_word] |= tail_mask;
        } else {
            data_[end_word] &= ~tail_mask;
        }
    }

    void set_bits(size_t start_index, uint64_t bits, uint8_t num_bits) override {
        if (num_bits == 0) return;
        if (num_bits > 64) throw std::invalid_argument("num_bits cannot exceed 64");
        if (start_index + num_bits > size_) throw std::out_of_range("set_bits out of bounds");

        const size_t w = start_index / 64;
        const uint32_t off = start_index % 64;

        if (off + num_bits <= 64) {
            uint64_t mask = (num_bits == 64) ? ~0ULL : ((1ULL << num_bits) - 1ULL);
            mask <<= off;
            uint64_t v = (bits << off) & mask;
            data_[w] = (data_[w] & ~mask) | v;
        } else {
            const uint32_t left = 64 - off;
            const uint32_t right = num_bits - left;
            uint64_t mask0 = ~0ULL << off;
            uint64_t v0 = (bits << off) & mask0;
            data_[w] = (data_[w] & ~mask0) | v0;
            uint64_t mask1 = (right == 64) ? ~0ULL : ((1ULL << right) - 1ULL);
            uint64_t v1 = (bits >> left) & mask1;
            data_[w+1] = (data_[w+1] & ~mask1) | v1;
        }
    }

    size_t size() const override {
        return size_;
    }

    // NOTE: Caller must call enable_rank()/enable_select*() before using these methods.
    // No runtime check is performed for performance.

    size_t rank(size_t pos) const override {
        return rank_support_->rank(pos);
    }

    size_t select(size_t k) const override {
        return select1_support_->select(k - 1);
    }

    size_t select0(size_t k) const override {
        return select0_support_->selectZero(k - 1);
    }

    // Aliases for consistency with code using unchecked variants
    size_t rank_unchecked(size_t pos) const { return rank_support_->rank(pos); }
    size_t select_unchecked(size_t k) const { return select1_support_->select(k - 1); }
    size_t select0_unchecked(size_t k) const { return select0_support_->selectZero(k - 1); }

    size_t size_in_bytes() const override {
        size_t total = data_.size() * sizeof(uint64_t);
        if (rank_support_.has_value()) {
            total += rank_support_->bitCount() / 8;
        }
        if (select1_support_.has_value()) {
            total += select1_support_->bitCount() / 8;
        }
        if (select0_support_.has_value()) {
            total += select0_support_->bitCount() / 8;
        }
        return total;
    }

    size_t support_size_in_bytes() const override {
        size_t total = 0;
        if (rank_support_.has_value()) {
            total += rank_support_->bitCount() / 8;
        }
        if (select1_support_.has_value()) {
            total += select1_support_->bitCount() / 8;
        }
        if (select0_support_.has_value()) {
            total += select0_support_->bitCount() / 8;
        }
        return total;
    }

    size_t size_in_megabytes() const override {
        return (size_in_bytes() + 1024 * 1024 - 1) / (1024 * 1024);
    }

    void serialize(std::ofstream& out) const override {
        if (!out) {
            throw std::runtime_error("Cannot write to stream");
        }
        out.write(reinterpret_cast<const char*>(&size_), sizeof(size_));
        size_t num_words = data_.size();
        out.write(reinterpret_cast<const char*>(&num_words), sizeof(num_words));
        out.write(reinterpret_cast<const char*>(data_.data()), num_words * sizeof(uint64_t));
    }

    static SUXBitVector load(std::ifstream& in) {
        if (!in) {
            throw std::runtime_error("Cannot open stream");
        }
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        size_t num_words;
        in.read(reinterpret_cast<char*>(&num_words), sizeof(num_words));
        std::vector<uint64_t> data(num_words);
        in.read(reinterpret_cast<char*>(data.data()), num_words * sizeof(uint64_t));
        return SUXBitVector(std::move(data), size);
    }

    static SUXBitVector load(const std::filesystem::path& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        return load(in);
    }

    std::unique_ptr<IBitVector> clone() const override {
        return std::make_unique<SUXBitVector>(*this);
    }

    // Key Optimization: In-place construction (emplace) instead of new/make_unique
    void enable_rank() override {
        if (!rank_support_.has_value()) {
            rank_support_.emplace(data_.data(), size_);
        }
    }

    void enable_select1() override {
        if (!select1_support_.has_value()) {
            select1_support_.emplace(data_.data(), size_);
        }
    }

    void enable_select0() override {
        if (!select0_support_.has_value()) {
            select0_support_.emplace(data_.data(), size_);
        }
    }

    uint64_t* raw_data_ptr() override {
        return data_.data();
    }
    
    const uint64_t* raw_data_ptr() const override {
        return data_.data();
    }
};

#endif