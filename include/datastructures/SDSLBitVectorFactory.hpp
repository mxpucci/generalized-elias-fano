#ifndef SDSLBITVECTORFACTORY_HPP
#define SDSLBITVECTORFACTORY_HPP

#include "IBitVectorFactory.hpp"
#include "SDSLBitVector.hpp"
#include <fstream>

/**
 * @brief Factory for creating SDSL-based bit vectors
 */
class SDSLBitVectorFactory : public IBitVectorFactory {
public:
    std::unique_ptr<IBitVector> create(size_t size) override {
        return std::make_unique<SDSLBitVector>(size);
    }
    
    std::unique_ptr<IBitVector> create(const std::vector<bool>& bits) override {
        return std::make_unique<SDSLBitVector>(bits);
    }
    
    std::unique_ptr<IBitVector> from_file(const std::filesystem::path& filepath) override {
        auto loaded = SDSLBitVector::load(filepath);
        return std::make_unique<SDSLBitVector>(std::move(loaded));
    }

    std::unique_ptr<IBitVector> from_stream(std::ifstream& in) override {
        auto loaded = SDSLBitVector::load(in);
        return std::make_unique<SDSLBitVector>(std::move(loaded));
    }

    double get_rank_overhead() const override {
        return SDSLBitVector::rank_overhead_per_bit();
    }

    double get_select1_overhead() const override {
        return SDSLBitVector::select1_overhead_per_bit();
    }

    double get_select0_overhead() const override {
        return SDSLBitVector::select0_overhead_per_bit();
    }
};

#endif 