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
#include <type_traits>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gef {

    #if __cplusplus < 202002L
    template<typename T>
    struct Span {
        const T* ptr;
        size_t n;
        Span(const T* p, size_t n) : ptr(p), n(n) {}
        const T* data() const { return ptr; }
        size_t size() const { return n; }
    };
    #endif


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
        if (k == 0) throw std::invalid_argument("Block size k cannot be zero.");

        const size_t num_partitions = (data.size() + k - 1) / k;
        m_partitions.resize(num_partitions);

        using PartitionView = Span<const T>;
        constexpr bool accepts_view_value      = std::is_constructible_v<Compressor, PartitionView, CompressorArgs...>;
        constexpr bool accepts_view_const_ref  = std::is_constructible_v<Compressor, const PartitionView&, CompressorArgs...>;
        constexpr bool accepts_view_ref        = std::is_constructible_v<Compressor, PartitionView&, CompressorArgs...>;
        constexpr bool accepts_vector_value    = std::is_constructible_v<Compressor, std::vector<T>, CompressorArgs...>;
        constexpr bool accepts_vector_constref = std::is_constructible_v<Compressor, const std::vector<T>&, CompressorArgs...>;
        constexpr bool accepts_vector_ref      = std::is_constructible_v<Compressor, std::vector<T>&, CompressorArgs...>;

        constexpr bool can_use_view   = accepts_view_value || accepts_view_const_ref || accepts_view_ref;
        constexpr bool can_use_vector = accepts_vector_value || accepts_vector_constref || accepts_vector_ref;

        static_assert(can_use_view || can_use_vector,
                      "Compressor must be constructible with Span<const T> or std::vector<T> when used by UniformedPartitioner");

    #ifdef _OPENMP
        // For uniform-sized partitions, use static scheduling without explicit chunk
        // This distributes work evenly with minimal overhead
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < num_partitions; ++p) {
                const size_t start = p * k;
                const size_t end   = std::min(start + k, data.size());
                const size_t len   = end - start;

                Span<const T> view(data.data() + start, len);
                if constexpr (can_use_view) {
                    m_partitions[p] = std::unique_ptr<IGEF<T>>(new Compressor(view, args...));
                } else if constexpr (accepts_vector_value) {
                    std::vector<T> buffer(view.data(), view.data() + view.size());
                    m_partitions[p] = std::unique_ptr<IGEF<T>>(new Compressor(std::move(buffer), args...));
                } else {
                    std::vector<T> buffer(view.data(), view.data() + view.size());
                    m_partitions[p] = std::unique_ptr<IGEF<T>>(new Compressor(buffer, args...));
                }
        }
    #else
        for (size_t p = 0; p < num_partitions; ++p) {
            const size_t start = p * k;
            const size_t end   = std::min(start + k, data.size());
            const size_t len   = end - start;

            Span<const T> view(data.data() + start, len);
            if constexpr (can_use_view) {
                m_partitions[p] = std::unique_ptr<IGEF<T>>(new Compressor(view, args...));
            } else if constexpr (accepts_vector_value) {
                std::vector<T> buffer(view.data(), view.data() + view.size());
                m_partitions[p] = std::unique_ptr<IGEF<T>>(new Compressor(std::move(buffer), args...));
            } else {
                std::vector<T> buffer(view.data(), view.data() + view.size());
                m_partitions[p] = std::unique_ptr<IGEF<T>>(new Compressor(buffer, args...));
            }
        }
    #endif
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
        // Note: std::unique_ptr overhead is runtime memory overhead, not serialized data size
        total_bytes += m_partitions.size() * sizeof(std::unique_ptr<IGEF<T>>);

        for (const auto& p : m_partitions) {
            total_bytes += p->size_in_bytes();
        }
        return total_bytes;
    }

    size_t theoretical_size_in_bytes() const override {
        size_t total_bytes = sizeof(m_original_size) + sizeof(m_block_size);
        size_t num_partitions = m_partitions.size();
        total_bytes += sizeof(num_partitions);
        // Note: std::unique_ptr overhead is runtime memory overhead, not serialized data size
        total_bytes += m_partitions.size() * sizeof(std::unique_ptr<IGEF<T>>);

        for (const auto& p : m_partitions) {
            total_bytes += p->theoretical_size_in_bytes();
        }
        return total_bytes;
    }

    std::vector<T> get_elements(size_t startIndex, size_t count) const override {
        std::vector<T> result;
        result.reserve(count);
        
        if (count == 0 || startIndex >= m_original_size) {
            return result;
        }
        
        const size_t endIndex = std::min(startIndex + count, m_original_size);
        
        // Identify which partitions we need to access
        const size_t start_partition = startIndex / m_block_size;
        const size_t end_partition = (endIndex - 1) / m_block_size;
        
        if (start_partition == end_partition) {
            // Fast path: all elements in same partition
            const size_t start_offset = startIndex % m_block_size;
            const size_t elements_to_get = endIndex - startIndex;
            auto partition_elements = m_partitions[start_partition]->get_elements(start_offset, elements_to_get);
            result.insert(result.end(), partition_elements.begin(), partition_elements.end());
        } else {
            const size_t num_partitions_spanned = end_partition - start_partition + 1;
            
            // Parallelize when spanning 2+ partitions (uniform workload justifies lower threshold)
            #ifdef _OPENMP
            if (num_partitions_spanned >= 2) {
                // Parallel approach: each thread fetches from its partition(s)
                // Then we combine results sequentially to maintain order
                std::vector<std::vector<T>> partition_results(num_partitions_spanned);
                
                // Use static scheduling for uniform-sized partitions (less overhead than dynamic)
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < num_partitions_spanned; ++i) {
                    const size_t p = start_partition + i;
                    const size_t partition_start = p * m_block_size;
                    const size_t partition_end = std::min(partition_start + m_block_size, m_original_size);
                    
                    const size_t range_start = std::max(startIndex, partition_start);
                    const size_t range_end = std::min(endIndex, partition_end);
                    
                    const size_t offset_in_partition = range_start - partition_start;
                    const size_t count_in_partition = range_end - range_start;
                    
                    partition_results[i] = m_partitions[p]->get_elements(offset_in_partition, count_in_partition);
                }
                
                // Combine results sequentially to maintain order
                for (const auto& partition_result : partition_results) {
                    result.insert(result.end(), partition_result.begin(), partition_result.end());
                }
            } else
            #endif
            {
                // Sequential fallback for small spans or no OpenMP
                for (size_t p = start_partition; p <= end_partition; ++p) {
                    const size_t partition_start = p * m_block_size;
                    const size_t partition_end = std::min(partition_start + m_block_size, m_original_size);
                    
                    const size_t range_start = std::max(startIndex, partition_start);
                    const size_t range_end = std::min(endIndex, partition_end);
                    
                    const size_t offset_in_partition = range_start - partition_start;
                    const size_t count_in_partition = range_end - range_start;
                    
                    auto partition_elements = m_partitions[p]->get_elements(offset_in_partition, count_in_partition);
                    result.insert(result.end(), partition_elements.begin(), partition_elements.end());
                }
            }
        }
        
        return result;
    }

    T operator[](size_t index) const override {
        if (index >= m_original_size) [[unlikely]] {
             throw std::out_of_range("index out of range in UniformedPartitioner");
        }
        
        // Division and modulo are expensive - but necessary here
        // Compiler will optimize to shift+mask if m_block_size is power of 2
        const size_t partition_index = index / m_block_size;
        const size_t index_in_partition = index % m_block_size;
        
        // Direct pointer dereference faster than at() (no bounds check)
        return (*m_partitions[partition_index])[index_in_partition];
    }

    uint8_t split_point() const override {
        return 0;
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Serialization
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    using IGEF<T>::serialize;
    void serialize(std::ofstream& ofs) const override {
        if (!ofs.is_open()) {
            throw std::runtime_error("Output file stream is not open for serialization.");
        }
        ofs.write(reinterpret_cast<const char*>(&m_original_size), sizeof(m_original_size));
        ofs.write(reinterpret_cast<const char*>(&m_block_size), sizeof(m_block_size));

        const size_t num_partitions = m_partitions.size();
        ofs.write(reinterpret_cast<const char*>(&num_partitions), sizeof(num_partitions));

        // Note: Serialization to a single stream must be sequential
        // Parallelizing would require writing to separate buffers then merging
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
