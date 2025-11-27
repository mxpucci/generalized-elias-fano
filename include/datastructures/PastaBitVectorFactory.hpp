#ifndef PASTABITVECTORFACTORY_HPP
#define PASTABITVECTORFACTORY_HPP

#include "IBitVectorFactory.hpp"
#include "PastaBitVector.hpp"
#include <fstream>

/**
 * @brief Factory for creating Pasta-based bit vectors
 */
class PastaBitVectorFactory : public IBitVectorFactory {
public:
    std::unique_ptr<IBitVector> create(size_t size) override {
        auto bv = std::make_unique<PastaBitVector>(size);
        initialize_support(*bv);
        return bv;
    }

    std::unique_ptr<IBitVector> create(const std::vector<bool>& bits) override {
        auto bv = std::make_unique<PastaBitVector>(bits);
        initialize_support(*bv);
        return bv;
    }

    std::unique_ptr<IBitVector> from_file(const std::filesystem::path& filepath) override {
        auto loaded = PastaBitVector::load(filepath);
        loaded.enable_rank();
        loaded.enable_select1();
        loaded.enable_select0();
        return std::make_unique<PastaBitVector>(std::move(loaded));
    }

    std::unique_ptr<IBitVector> from_stream(std::ifstream& in) override {
        auto loaded = PastaBitVector::load(in);
        loaded.enable_rank();
        loaded.enable_select1();
        loaded.enable_select0();
        return std::make_unique<PastaBitVector>(std::move(loaded));
    }

    double get_rank_overhead() const override {
        return PastaBitVector::rank_overhead_per_bit();
    }

    double get_select1_overhead() const override {
        return PastaBitVector::select1_overhead_per_bit();
    }

    double get_select0_overhead() const override {
        return PastaBitVector::select0_overhead_per_bit();
    }

private:
    static void initialize_support(PastaBitVector& bv) {
        bv.enable_rank();
        bv.enable_select1();
        bv.enable_select0();
    }
};

#endif


