//
// Created by Michelangelo Pucci on 17/07/25.
//

#ifndef UNIFORMED_PARTITIONER_HPP
#define UNIFORMED_PARTITIONER_HPP

#include "IGEF.hpp"
#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace gef {

/**
 * @brief A class that partitions a sequence and applies a given compressor to each partition.
 *
 * This class acts as a wrapper around another IGEF-compliant compressor. It takes a large
 * vector of data, splits it into smaller blocks of a specified size `k`, and then uses
 * the provided `Compressor` class to compress each block independently.
 *
 * This is useful for applying compressors that are efficient on smaller data sizes to a large
 * dataset, or to enable parallel processing on blocks.
 *
 * @tparam T The integral type of the data elements.
 * @tparam Compressor The IGEF-compliant compressor class to use for each partition.
 * @tparam CompressorArgs The types of additional arguments to pass to the compressor's constructor.
 */
#if __cplusplus >= 202002L
template<IntegralType T, class Compressor, typename... CompressorArgs>
#else
template<typename T, class Compressor, typename... CompressorArgs>
#endif
class UniformedPartitioner : public IGEF<T> {
    static_assert(std::is_base_of_v<IGEF<T>, Compressor>, "Compressor must be a subclass of IGEF<T>");

public:
    /**
     * @brief Constructs a UniformedPartitioner by compressing data in blocks.
     * @param data The input vector to compress.
     * @param k The size of each block. The last block may be smaller.
     * @param args Additional arguments to be forwarded to the Compressor's constructor for each block.
     */
    UniformedPartitioner(const std::vector<T>& data, size_t k, CompressorArgs... args)
        : m_original_size(data.size()), m_block_size(k) {
        if (k == 0) {
            throw std::invalid_argument("Block size k cannot be zero.");
        }

        m_partitions.reserve((data.size() + k - 1) / k);
        for (size_t i = 0; i < data.size(); i += k) {
            auto start_it = data.begin() + i;
            auto end_it = data.begin() + std::min(i + k, data.size());

            // This assumes the Compressor can be constructed from a vector.
            std::vector<T> chunk(start_it, end_it);
            m_partitions.emplace_back(std::make_unique<Compressor>(chunk, args...));
        }
    }

    /**
     * @brief Default constructor. Used for loading from a stream.
     */
    UniformedPartitioner() : m_original_size(0), m_block_size(0) {}

    ~UniformedPartitioner() override = default;

    UniformedPartitioner(const UniformedPartitioner&) = delete;
    UniformedPartitioner& operator=(const UniformedPartitioner&) = delete;
    UniformedPartitioner(UniformedPartitioner&&) = default;
    UniformedPartitioner& operator=(UniformedPartitioner&&) = default;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // IGEF interface implementation
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    size_t size() const override {
        return m_original_size;
    }

    size_t size_in_bytes() const override {
        size_t total_bytes = sizeof(m_original_size) + sizeof(m_block_size);

        // Store number of partitions to facilitate loading
        size_t num_partitions = m_partitions.size();
        total_bytes += sizeof(num_partitions);

        for (const auto& p : m_partitions) {
            total_bytes += p->size_in_bytes();
        }
        return total_bytes;
    }

    size_t theoretical_size_in_bytes() const override {
        size_t total_bytes = sizeof(m_original_size) + sizeof(m_block_size);
        size_t num_partitions = m_partitions.size();
        total_bytes += sizeof(num_partitions);

        for (const auto& p : m_partitions) {
            total_bytes += p->theoretical_size_in_bytes();
        }
        return total_bytes;
    }

    T operator[](size_t index) const override {
        if (index >= m_original_size) {
             // Following at() convention, though operator[] is not always guaranteed to check.
             throw std::out_of_range("index out of range in UniformedPartitioner");
        }
        size_t partition_index = index / m_block_size;
        size_t index_in_partition = index % m_block_size;
        return (*m_partitions.at(partition_index))[index_in_partition];
    }

    uint8_t split_point() const override {
        return 0;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Serialization
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    void serialize(std::ofstream& ofs) const override {
        if (!ofs.is_open()) {
            throw std::runtime_error("Output file stream is not open for serialization.");
        }
        ofs.write(reinterpret_cast<const char*>(&m_original_size), sizeof(m_original_size));
        ofs.write(reinterpret_cast<const char*>(&m_block_size), sizeof(m_block_size));

        size_t num_partitions = m_partitions.size();
        ofs.write(reinterpret_cast<const char*>(&num_partitions), sizeof(num_partitions));

        for (const auto& p : m_partitions) {
            p->serialize(ofs);
        }
    }

    void load(std::ifstream& ifs, const std::shared_ptr<IBitVectorFactory> bit_vector_factory) override {
        if (!ifs.is_open()) {
            throw std::runtime_error("Input file stream is not open for loading.");
        }
        ifs.read(reinterpret_cast<char*>(&m_original_size), sizeof(m_original_size));
        ifs.read(reinterpret_cast<char*>(&m_block_size), sizeof(m_block_size));

        if (ifs.fail() || m_block_size == 0) {
            throw std::runtime_error("Failed to read or invalid data from stream during UniformedPartitioner load.");
        }

        size_t num_partitions;
        ifs.read(reinterpret_cast<char*>(&num_partitions), sizeof(num_partitions));
        if (ifs.fail()) {
            throw std::runtime_error("Failed to read number of partitions from stream.");
        }

        m_partitions.clear();
        m_partitions.reserve(num_partitions);
        for (size_t i = 0; i < num_partitions; ++i) {
            // This requires Compressor to be default-constructible for loading.
            auto partition = std::make_unique<Compressor>();
            partition->load(ifs, bit_vector_factory);
            m_partitions.push_back(std::move(partition));
        }
    }

private:
    std::vector<std::unique_ptr<IGEF<T>>> m_partitions;
    size_t m_original_size;
    size_t m_block_size;
};

} // namespace gef

#endif // UNIFORMED_PARTITIONER_HPP
