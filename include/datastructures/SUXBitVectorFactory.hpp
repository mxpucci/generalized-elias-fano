#ifndef SUXBITVECTORFACTORY_HPP
#define SUXBITVECTORFACTORY_HPP

#include "IBitVectorFactory.hpp"
#include "SUXBitVector.hpp"
#include <fstream>

/**
 * @brief Factory for creating SUX-based bit vectors
 */
class SUXBitVectorFactory : public IBitVectorFactory {
public:
    std::unique_ptr<IBitVector> create(size_t size) override {
        return std::make_unique<SUXBitVector>(size);
    }
    
    std::unique_ptr<IBitVector> create(const std::vector<bool>& bits) override {
        return std::make_unique<SUXBitVector>(bits);
    }
    
    std::unique_ptr<IBitVector> from_file(const std::filesystem::path& filepath) override {
        auto loaded = SUXBitVector::load(filepath);
        return std::make_unique<SUXBitVector>(std::move(loaded));
    }

    std::unique_ptr<IBitVector> from_stream(std::ifstream& in) override {
        auto loaded = SUXBitVector::load(in);
        return std::make_unique<SUXBitVector>(std::move(loaded));
    }

    double get_rank_overhead() const override {
        return SUXBitVector::rank_overhead_per_bit();
    }

    double get_select1_overhead() const override {
        return SUXBitVector::select1_overhead_per_bit();
    }

    double get_select0_overhead() const override {
        return SUXBitVector::select0_overhead_per_bit();
    }
};

#endif

