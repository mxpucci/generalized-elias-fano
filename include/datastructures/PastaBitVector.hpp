#ifndef PASTABITVECTOR_HPP
#define PASTABITVECTOR_HPP

#include "IBitVector.hpp"

#include <pasta/bit_vector/bit_vector.hpp>
#include <pasta/bit_vector/support/flat_rank_select.hpp>

#include <algorithm>
#include <vector>
#include <memory>
#include <optional>
#include <fstream>
#include <filesystem>
#include <iostream>

/**
 * @brief Pasta-based implementation of IBitVector
 * Uses the Pasta toolbox rank/select structures.
 *
 * Default: flat-popcount (fast rank & select, low overhead).
 * Reference: Kurpicz, "Engineering Compact Data Structures for Rank and Select Queries on Bit Vectors"
 * :contentReference[oaicite:1]{index=1}
 */
class PastaBitVector : public IBitVector {
private:
    using bv_type = pasta::BitVector;

    // Use pasta's optimized FlatRankSelect support
    using support_type = pasta::FlatRankSelect<>;

    bv_type bv_;
    mutable std::unique_ptr<support_type> support_;

public:
    static constexpr bool reverse_bit_order = false;
    static double rank_overhead_per_bit() { return 0.0358; }
    static double select1_overhead_per_bit() { return 0.0358; }
    static double select0_overhead_per_bit() { return 0.0358; }


    using IBitVector::rank0;
    using IBitVector::serialize;
    // ---- Constructors ----

    explicit PastaBitVector(size_t size)
        : bv_(size) {}

    explicit PastaBitVector(const std::vector<bool>& bits)
        : bv_(bits.size()) {
        for (size_t i = 0; i < bits.size(); ++i) {
            bv_[i] = bits[i];
        }
    }

    explicit PastaBitVector(bv_type bv)
        : bv_(std::move(bv)) {}

    // ---- Rule of 5 ----

    ~PastaBitVector() = default;

    PastaBitVector(const PastaBitVector& other)
        : bv_(other.bv_.size()) {

        // Manually copy underlying data because pasta::BitVector copy ctor is deleted
        auto src_span = other.bv_.data();
        auto dest_span = bv_.data();
        std::copy(src_span.begin(), src_span.end(), dest_span.begin());

        if (other.support_) {
            support_ = std::make_unique<support_type>(bv_);
        }
    }

    PastaBitVector& operator=(const PastaBitVector& other) {
        if (this != &other) {
            // Manually copy underlying data because pasta::BitVector copy ctor is deleted
            bv_type temp_copy(other.bv_.size());
            auto src_span = other.bv_.data();
            auto dest_span = temp_copy.data();
            std::copy(src_span.begin(), src_span.end(), dest_span.begin());

            bv_ = std::move(temp_copy);

            // Rebuild support structure
            support_.reset();
            if (other.support_) {
                support_ = std::make_unique<support_type>(bv_);
            }
        }
        return *this;
    }

    PastaBitVector(PastaBitVector&& other) noexcept
        : bv_(std::move(other.bv_)) {

        if (other.support_) {
            support_ = std::make_unique<support_type>(bv_);
        }
        other.support_.reset();
    }

    PastaBitVector& operator=(PastaBitVector&& other) noexcept {
        if (this != &other) {
            bv_ = std::move(other.bv_);
            support_.reset();

            if (other.support_) {
                support_ = std::make_unique<support_type>(bv_);
            }
            other.support_.reset();
        }
        return *this;
    }

    // ---- Bit Access ----

    bool operator[](size_t index) const override {
        return bv_[index];
    }

    auto operator[](size_t index) -> decltype(bv_[index]) {
        return bv_[index];
    }

    void set(size_t index, bool value) override {
        bv_[index] = value;
    }

    // ---- Range Operations ----

    void set_range(size_t start, size_t count, bool value) override {
        for (size_t i = 0; i < count; ++i) {
            bv_[start + i] = value;
        }
    }

    inline void set_bits(size_t start_index, uint64_t bits, uint8_t num_bits) override {
        for (uint8_t i = 0; i < num_bits; ++i) {
            bv_[start_index + i] = (bits >> i) & 1;
        }
    }

    // ---- Sizes ----

    size_t size() const override {
        return bv_.size();
    }

    size_t size_in_bytes() const override {
        size_t total = bv_.space_usage();
        if (support_) {
            total += support_->space_usage();
        }
        return total;
    }

    size_t support_size_in_bytes() const override {
        if (!support_) return 0;
        return support_->space_usage();
    }

    size_t size_in_megabytes() const override {
        return (size_in_bytes() + 1024 * 1024 - 1) / (1024 * 1024);
    }

    // ---- Rank & Select ----

    size_t rank(size_t pos) const override {
        if (!support_)
            throw std::runtime_error("Rank support not enabled");
        return support_->rank1(pos);
    }

    size_t select(size_t k) const override {
        if (!support_)
            throw std::runtime_error("Select support not enabled");
        return support_->select1(k);
    }

    size_t select0(size_t k) const override {
        if (!support_)
            throw std::runtime_error("Select support not enabled");
        return support_->select0(k);
    }

    // ---- Support Construction ----

    void enable_rank() override {
        if (!support_) {
            support_ = std::make_unique<support_type>(bv_);
        }
    }

    void enable_select1() override {
        if (!support_) {
            support_ = std::make_unique<support_type>(bv_);
        }
    }

    void enable_select0() override {
        if (!support_) {
            support_ = std::make_unique<support_type>(bv_);
        }
    }

    // ---- Serialization ----
    void serialize(std::ofstream& out) const override {
        // Serialize the size first
        size_t bit_size = bv_.size();
        out.write(reinterpret_cast<const char*>(&bit_size), sizeof(size_t));

        // Serialize the bits one by one (simpler approach)
        for (size_t i = 0; i < bit_size; ++i) {
            bool bit = bv_[i];
            out.write(reinterpret_cast<const char*>(&bit), sizeof(bool));
        }

        // Serialize whether we have support
        bool has_support = support_ != nullptr;
        out.write(reinterpret_cast<const char*>(&has_support), sizeof(bool));
    }

    static PastaBitVector load(std::ifstream& in) {
        if (!in) throw std::runtime_error("Cannot open stream");

        // Load the size
        size_t bit_size;
        in.read(reinterpret_cast<char*>(&bit_size), sizeof(size_t));

        // Create the bit vector
        bv_type bv(bit_size);

        // Load the bits one by one
        for (size_t i = 0; i < bit_size; ++i) {
            bool bit;
            in.read(reinterpret_cast<char*>(&bit), sizeof(bool));
            bv[i] = bit;
        }

        // Load support flag
        bool has_support = false;
        in.read(reinterpret_cast<char*>(&has_support), sizeof(bool));

        PastaBitVector result(std::move(bv));
        if (has_support) {
            result.enable_rank();
        }

        return result;
    }

    static PastaBitVector load(const std::filesystem::path& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        return load(in);
    }

    std::unique_ptr<IBitVector> clone() const override {
        return std::make_unique<PastaBitVector>(*this);
    }

    // ---- Raw access ----
    const uint64_t* raw_data_ptr() const {
        return bv_.data().data();
    }

    uint64_t* raw_data_ptr() {
        return bv_.data().data();
    }
};

#endif
