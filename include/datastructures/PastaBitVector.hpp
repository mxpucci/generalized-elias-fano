#ifndef PASTABITVECTOR_HPP
#define PASTABITVECTOR_HPP

#include "IBitVector.hpp"
#include "PastaOptimizedRankSelect.hpp"
#include <pasta/bit_vector/bit_vector.hpp>
#include <pasta/bit_vector/support/flat_rank_select.hpp>
#include <algorithm>
#include <vector>
#include <memory>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>

/**
 * @brief Parametric BitVector wrapper for PASTA Toolbox.
 *
 * @tparam OptRank1   Enable optimization for Rank(1)
 * @tparam OptSelect1 Enable optimization for Select(1)
 * @tparam OptSelect0 Enable optimization for Select(0)
 *
 * Paper Reference: "Engineering Compact Data Structures for Rank and Select Queries on Bit Vectors"
 * - Disabling unused supports reduces memory overhead.
 * - Enabling OptSelect0 prevents the ~1.6x slowdown observed in generic structures.
 */
template <bool OptRank1 = true, bool OptSelect1 = true, bool OptSelect0 = true>
class PastaBitVectorT : public IBitVector {
private:
    using bv_type = pasta::BitVector;

    // Configure FlatRankSelect at compile time.
    // Params: <Rank1, Rank0, Select1, Select0>
    // We keep Rank0 false as it is rarely used in GEF/Elias-Fano.
    // Rank1 is almost always required as a prerequisite for rank-based Select.
    static constexpr pasta::OptimizedFor opt_target = 
        (OptSelect0 && !OptSelect1) ? pasta::OptimizedFor::ZERO_QUERIES : 
        (!OptSelect0 && !OptSelect1) ? pasta::OptimizedFor::DONT_CARE : // Case for RankOnly
        pasta::OptimizedFor::ONE_QUERIES;

    // Use INTRINSICS only if on x86_64 AND SIMD is not disabled
    #if (defined(__x86_64__) || defined(_M_X64)) && !defined(GEF_DISABLE_SIMD)
        static constexpr pasta::FindL2FlatWith search_strategy = pasta::FindL2FlatWith::INTRINSICS;
    #else
        static constexpr pasta::FindL2FlatWith search_strategy = pasta::FindL2FlatWith::LINEAR_SEARCH;
    #endif

    using support_type = OptimizedFlatRankSelect<
        opt_target, 
        search_strategy, 
        bv_type
    >; 
    bv_type bv_;
    mutable std::unique_ptr<support_type> support_;

public:
    static constexpr bool reverse_bit_order = false;

    // Overhead depends on what is enabled. 
    // Base overhead for Flat-Popcount is ~3.58%[cite: 287].
    // Disabling Select1/Select0 samples reduces this further.
    static double rank_overhead_per_bit() { return OptRank1 ? 0.0358 : 0.0; }
    static double select1_overhead_per_bit() { return OptSelect1 ? 0.0358 : 0.0; }
    static double select0_overhead_per_bit() { return OptSelect0 ? 0.0358 : 0.0; }

    // ---- Constructors ----

    explicit PastaBitVectorT(size_t size)
        : bv_(size, false) {}

    explicit PastaBitVectorT(const std::vector<bool>& bits)
        : bv_(bits.size(), false) {
        for (size_t i = 0; i < bits.size(); ++i) {
            bv_[i] = bits[i];
        }
    }

    explicit PastaBitVectorT(bv_type bv)
        : bv_(std::move(bv)) {}

    // ---- Rule of 5 ----

    ~PastaBitVectorT() = default;

    PastaBitVectorT(const PastaBitVectorT& other)
        : bv_(other.bv_.size()) {

        // Manually copy underlying data because pasta::BitVector copy ctor is deleted
        auto src_span = other.bv_.data();
        auto dest_span = bv_.data();
        std::copy(src_span.begin(), src_span.end(), dest_span.begin());

        if (other.support_) {
            support_ = std::make_unique<support_type>(bv_);
        }
    }

    PastaBitVectorT& operator=(const PastaBitVectorT& other) {
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

    PastaBitVectorT(PastaBitVectorT&& other) noexcept
        : bv_(std::move(other.bv_)) {

        if (other.support_) {
            support_ = std::make_unique<support_type>(bv_);
        }
        other.support_.reset();
    }

    PastaBitVectorT& operator=(PastaBitVectorT&& other) noexcept {
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

    inline bool operator[](size_t index) const override { return bv_[index]; }
    inline auto operator[](size_t index) { return bv_[index]; }
    inline void set(size_t index, bool value) override { bv_[index] = value; }

    inline void set_range(size_t start, size_t count, bool value) override {
        for (size_t i = 0; i < count; ++i) bv_[start + i] = value;
    }

    inline void set_bits(size_t start, uint64_t bits, uint8_t n) override {
        for (uint8_t i = 0; i < n; ++i) bv_[start + i] = (bits >> i) & 1;
    }

    // ---- Raw Access for FastBitWriter ----
    
    const uint64_t* raw_data_ptr() const { return bv_.data().data(); }
    uint64_t* raw_data_ptr() { return bv_.data().data(); }

    // ---- Rank & Select ----

    inline size_t rank(size_t pos) const override {
        if constexpr (OptRank1) return support_->rank1(pos);
        else throw std::runtime_error("Rank1 not enabled at compile time");
    }

    inline size_t select(size_t k) const override {
        if constexpr (OptSelect1) return support_->select1(k);
        else throw std::runtime_error("Select1 not enabled at compile time");
    }

    inline size_t select0(size_t k) const override {
        if constexpr (OptSelect0) return support_->select0(k);
        else throw std::runtime_error("Select0 not enabled at compile time");
    }

    // Unchecked variants for hot paths
    inline size_t rank_unchecked(size_t pos) const { return support_->rank1(pos); }
    inline size_t select_unchecked(size_t k) const { return support_->select1(k); }
    inline size_t select0_unchecked(size_t k) const { return support_->select0(k); }

    // ---- Support Construction ----

    void enable_rank() override {
        if constexpr (OptRank1) {
            if (!support_) support_ = std::make_unique<support_type>(bv_);
        }
    }

    void enable_select1() override {
        if constexpr (OptSelect1) {
            if (!support_) support_ = std::make_unique<support_type>(bv_);
        }
    }

    void enable_select0() override {
        if constexpr (OptSelect0) {
            if (!support_) support_ = std::make_unique<support_type>(bv_);
        }
    }

    // ---- Sizes ----

    size_t size() const override { return bv_.size(); }
    
    size_t size_in_bytes() const override {
        return bv_.space_usage() + (support_ ? support_->space_usage() : 0);
    }
    
    size_t support_size_in_bytes() const override {
        return support_ ? support_->space_usage() : 0;
    }

    size_t size_in_megabytes() const override {
        return (size_in_bytes() + 1024 * 1024 - 1) / (1024 * 1024);
    }

    // ---- Serialization ----

    using IBitVector::serialize;

    void serialize(std::ofstream& out) const override {
        size_t sz = bv_.size();
        out.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
        bool has_supp = (support_ != nullptr);
        out.write(reinterpret_cast<const char*>(&has_supp), sizeof(bool));
        
        // Bulk write bits
        auto data = bv_.data();
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * 8);
    }

    static PastaBitVectorT load(std::ifstream& in) {
        size_t sz;
        in.read(reinterpret_cast<char*>(&sz), sizeof(size_t));
        bool has_supp;
        in.read(reinterpret_cast<char*>(&has_supp), sizeof(bool));
        
        PastaBitVectorT res(sz);
        auto data = res.bv_.data();
        in.read(reinterpret_cast<char*>(data.data()), data.size() * 8);

        if (has_supp) {
            // Rebuilds specific supported structure
            res.support_ = std::make_unique<support_type>(res.bv_);
        }
        return res;
    }

    static PastaBitVectorT load(const std::filesystem::path& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Could not open file: " + filepath.string());
        }
        return load(in);
    }

    std::unique_ptr<IBitVector> clone() const override {
        return std::make_unique<PastaBitVectorT>(*this);
    }
};

// =========================================================
// Type Aliases for Common Use Cases
// =========================================================

// 1. General Purpose (All features enabled)
using PastaBitVector = PastaBitVectorT<true, true, true>;

// 2. Select-0 Optimized (For Gap Encoding / B_STAR_GEF)
//    Enables Rank1 (needed for internal select logic) and Select0. 
//    Disables Select1 index to save space.
using PastaGapBitVector = PastaBitVectorT<true, false, true>;

// 3. Rank Only (For L vector in rare cases or frequency arrays)
//    Corresponds to "Wide-Popcount" or "Flat-Popcount" without samples[cite: 228].
using PastaRankBitVector = PastaBitVectorT<true, false, false>;

// 4. Exception BitVector (For B vector in GEF)
//    Enables Rank1 and Select1 (to locate exceptions).
//    Disables Select0 to save space.
using PastaExceptionBitVector = PastaBitVectorT<true, true, false>;

#endif