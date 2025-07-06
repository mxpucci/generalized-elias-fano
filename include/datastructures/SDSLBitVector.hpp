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
        : bv_(std::move(bv)) {
    }

    SDSLBitVector(size_t size)
        : SDSLBitVector(sdsl::bit_vector(size, 0)) {}

    SDSLBitVector(const std::vector<bool>& bits) {
        bv_ = sdsl::bit_vector(bits.size());
        for (size_t i = 0; i < bits.size(); ++i) {
            bv_[i] = bits[i];
        }
    }

    // Rule of 5: Destructor
    ~SDSLBitVector() = default;

    // Rule of 5: Copy constructor
    SDSLBitVector(const SDSLBitVector& other) : bv_(other.bv_) {
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
    SDSLBitVector(SDSLBitVector&& other) noexcept : bv_(std::move(other.bv_)) {
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