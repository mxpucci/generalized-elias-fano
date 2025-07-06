#ifndef IBITVECTORFACTORY_HPP
#define IBITVECTORFACTORY_HPP

#include "IBitVector.hpp"
#include <memory>
#include <vector>
#include <filesystem>

/**
 * @brief Factory interface for creating bit vectors
 */
class IBitVectorFactory {
public:
    virtual ~IBitVectorFactory() = default;
    
    virtual std::unique_ptr<IBitVector> create(size_t size) = 0;
    virtual std::unique_ptr<IBitVector> create(const std::vector<bool>& bits) = 0;
    
    virtual std::unique_ptr<IBitVector> from_file(const std::filesystem::path& filepath) = 0;
    virtual std::unique_ptr<IBitVector> from_stream(std::ifstream& in) = 0;
};

#endif 