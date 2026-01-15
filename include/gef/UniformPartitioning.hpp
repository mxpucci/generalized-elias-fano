//
// Created by Michelangelo Pucci on 17/07/25.
//

#ifndef UNIFORM_PARTITIONING_HPP
#define UNIFORM_PARTITIONING_HPP

#include "IGEF.hpp"
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <optional>

#if defined(_OPENMP) && !defined(GEF_DISABLE_OPENMP)
#include <omp.h>
#define GEF_USE_OPENMP 1
#endif

namespace gef {

/**
 * @brief A class that partitions a sequence and applies a given compressor to each partition.
 *
 * This class acts as a wrapper around another IGEF-compliant compressor. It takes a large
 * vector of data, splits it into smaller blocks of a specified size `K`, and then uses
 * the provided `Compressor` class to compress each block independently.
 *
 * This is useful for applying compressors that are efficient on smaller data sizes to a large
 * dataset, or to enable parallel processing on blocks.
 *
 * @tparam T The integral type of the data elements.
 * @tparam Compressor The IGEF-compliant compressor class to use for each partition.
 * @tparam K The size of each block.
 * @tparam CompressorArgs The types of additional arguments to pass to the compressor's constructor.
 */
#if __cplusplus >= 202002L
template<IntegralType T, class Compressor, size_t K, typename... CompressorArgs>
#else
template<typename T, class Compressor, size_t K, typename... CompressorArgs>
#endif
class UniformPartitioning : public IGEF<T> {
    static_assert(std::is_base_of_v<IGEF<T>, Compressor>, "Compressor must be a subclass of IGEF<T>");
    static_assert(K > 0, "Block size K cannot be zero.");

public:
    constexpr static size_t MIN_ELEMENTS_PER_THREAD = 100000;

    /**
     * @brief Constructs a UniformPartitioning by compressing data in blocks.
     * @param data The input vector to compress.
     * @param args Additional arguments to be forwarded to the Compressor's constructor for each block.
     */
    UniformPartitioning(const std::vector<T>& data, CompressorArgs... args)
    : m_original_size(data.size()) {
        const size_t num_partitions = (data.size() + K - 1) / K;
        m_partitions.reserve(num_partitions);

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
                      "Compressor must be constructible with Span<const T> or std::vector<T> when used by UniformPartitioning");

        // Sequential implementation - used when OpenMP is disabled
        auto build_sequential = [&]() {
            m_partitions.resize(num_partitions);
            
            // Single-threaded: use indexed construction via emplace()
            // Since m_partitions is vector<optional<Compressor>>, resize() is cheap
            // (just creates empty optionals, no Compressor construction)
            if constexpr (can_use_view) {
                for (size_t p = 0; p < num_partitions; ++p) {
                    const size_t start = p * K;
                    const size_t end   = std::min(start + K, data.size());
                    const size_t len   = end - start;
                    m_partitions[p].emplace(PartitionView(data.data() + start, len), args...);
                }
            } else if constexpr (!accepts_vector_value) {
                // Re-use buffer optimization
                std::vector<T> buffer;
                buffer.reserve(K);
                for (size_t p = 0; p < num_partitions; ++p) {
                    const size_t start = p * K;
                    const size_t end   = std::min(start + K, data.size());
                    const size_t len   = end - start;
                    buffer.assign(data.data() + start, data.data() + start + len);
                    m_partitions[p].emplace(buffer, args...);
                }
            } else {
                // Must pass by value (move)
                for (size_t p = 0; p < num_partitions; ++p) {
                    const size_t start = p * K;
                    const size_t end   = std::min(start + K, data.size());
                    const size_t len   = end - start;
                    PartitionView view(data.data() + start, len);
                    std::vector<T> buffer(view.data(), view.data() + view.size());
                    m_partitions[p].emplace(std::move(buffer), args...);
                }
            }
        };

    #if GEF_USE_OPENMP
        m_partitions.resize(num_partitions);
        
        
        const size_t elements_limit = std::max<size_t>(1, data.size() / MIN_ELEMENTS_PER_THREAD);
        
        int max_threads = omp_get_max_threads();
        int n_threads = std::max(1, std::min({
            max_threads, 
            static_cast<int>(num_partitions), 
            static_cast<int>(elements_limit)
        }));
        #pragma omp parallel num_threads(n_threads)
        {
            // Thread-local buffer to avoid repeated allocations in fallback cases
            std::vector<T> buffer;
            if constexpr (!can_use_view && !accepts_vector_value) {
                buffer.reserve(K);
            }

            // Use dynamic scheduling for better load balancing
            // Chunk size of 16 provides good balance between scheduling overhead and load distribution
            // Threads that finish early can pick up more work, improving overall parallel efficiency
            #pragma omp for schedule(dynamic, 16)
            for (size_t p = 0; p < num_partitions; ++p) {
                const size_t start = p * K;
                const size_t end   = std::min(start + K, data.size());
                const size_t len   = end - start;

                Span<const T> view(data.data() + start, len);
                if constexpr (can_use_view) {
                    m_partitions[p].emplace(view, args...);
                } else if constexpr (accepts_vector_value) {
                    std::vector<T> move_buffer(view.data(), view.data() + view.size());
                    m_partitions[p].emplace(std::move(move_buffer), args...);
                } else {
                    buffer.assign(view.data(), view.data() + view.size());
                    m_partitions[p].emplace(buffer, args...);
                }
            }
        }
        
        // Verify all partitions are initialized after parallel construction
        for (size_t p = 0; p < num_partitions; ++p) {
            if (!m_partitions[p].has_value()) {
                throw std::runtime_error("UniformPartitioning: partition " + std::to_string(p) + 
                    " was not initialized after parallel construction");
            }
        }
    #else
        build_sequential();
    #endif
    }


    /**
     * @brief Default constructor. Used for loading from a stream.
     */
    UniformPartitioning() : m_original_size(0) {}

    ~UniformPartitioning() override = default;

    UniformPartitioning(const UniformPartitioning&) = delete;
    UniformPartitioning& operator=(const UniformPartitioning&) = delete;
    UniformPartitioning(UniformPartitioning&&) = default;
    UniformPartitioning& operator=(UniformPartitioning&&) = default;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // IGEF interface implementation
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    size_t size() const override {
        return m_original_size;
    }

    size_t size_in_bytes() const override {
        size_t total_bytes = sizeof(m_original_size); // K is template param, not stored
        size_t num_partitions = m_partitions.size();
        total_bytes += sizeof(num_partitions);
        for (size_t i = 0; i < num_partitions; ++i) {
            total_bytes += partition(i).size_in_bytes();
        }
        return total_bytes;
    }

    size_t theoretical_size_in_bytes() const override {
        size_t total_bytes = sizeof(m_original_size);
        size_t num_partitions = m_partitions.size();
        total_bytes += sizeof(num_partitions);

        for (size_t i = 0; i < num_partitions; ++i) {
            total_bytes += partition(i).theoretical_size_in_bytes();
        }
        return total_bytes;
    }

    size_t get_elements(size_t startIndex, size_t count, std::vector<T>& output) const override {
        if (count == 0 || startIndex >= m_original_size) {
            return 0;
        }
        if (output.size() < count) {
            throw std::invalid_argument("output buffer is smaller than requested count");
        }
        
        const size_t endIndex = std::min(startIndex + count, m_original_size);
        const size_t total_requested = endIndex - startIndex;
        
        // Identify which partitions we need to access
        const size_t start_partition = startIndex / K;
        const size_t end_partition = (endIndex - 1) / K;
        
        auto fetch_partition_range = [&](size_t partition_index,
                                         size_t range_start,
                                         size_t range_end,
                                         std::vector<T>& buffer) -> size_t {
            const size_t partition_start = partition_index * K;
            const size_t offset_in_partition = range_start - partition_start;
            const size_t count_in_partition = range_end - range_start;
            if (count_in_partition == 0) {
                buffer.clear();
                return 0;
            }
            buffer.assign(count_in_partition, T{});
            size_t written = partition(partition_index).get_elements(
                offset_in_partition, count_in_partition, buffer);
            buffer.resize(written);
            return written;
        };
        
        // Fast path: all elements in same partition
        if (start_partition == end_partition) {
            std::vector<T> partition_buffer(total_requested);
            const size_t written = partition(start_partition).get_elements(
                startIndex % K, total_requested, partition_buffer);
            std::copy_n(partition_buffer.begin(), written, output.begin());
            return written;
        }
        
        size_t total_written = 0;
        const size_t num_partitions_spanned = end_partition - start_partition + 1;
        
        // Parallelize only when the overhead is justified:
        // - Need enough total elements to decompress (amortize thread overhead)
        // - Need enough partitions to distribute work effectively
        #if GEF_USE_OPENMP
        constexpr size_t MIN_ELEMENTS_FOR_PARALLEL_DECOMPRESS = 100000;
        constexpr size_t MIN_PARTITIONS_FOR_PARALLEL = 4;
        const bool use_parallel = (total_requested >= MIN_ELEMENTS_FOR_PARALLEL_DECOMPRESS) &&
                                  (num_partitions_spanned >= MIN_PARTITIONS_FOR_PARALLEL);
        
        if (use_parallel) {
            std::vector<std::vector<T>> partition_results(num_partitions_spanned);
            std::vector<size_t> partition_written(num_partitions_spanned, 0);
            
            const size_t elements_limit = std::max<size_t>(1, total_requested / MIN_ELEMENTS_PER_THREAD);
            
            int max_threads = omp_get_max_threads();
            int n_threads = std::max(1, std::min({
                max_threads, 
                static_cast<int>(num_partitions_spanned), 
                static_cast<int>(elements_limit)
            }));

            #pragma omp parallel for schedule(static) num_threads(n_threads)
            for (size_t i = 0; i < num_partitions_spanned; ++i) {
                const size_t p = start_partition + i;
                const size_t partition_start = p * K;
                const size_t partition_end = std::min(partition_start + K, m_original_size);
                const size_t range_start = std::max(startIndex, partition_start);
                const size_t range_end = std::min(endIndex, partition_end);
                partition_written[i] = fetch_partition_range(p, range_start, range_end, partition_results[i]);
            }
            
            for (size_t i = 0; i < num_partitions_spanned; ++i) {
                if (partition_written[i] == 0) {
                    continue;
                }
                std::copy_n(partition_results[i].begin(),
                            partition_written[i],
                            output.begin() + total_written);
                total_written += partition_written[i];
            }
            return total_written;
        }
        #endif
        
        // Sequential fallback for small spans or no OpenMP
        for (size_t p = start_partition; p <= end_partition; ++p) {
            const size_t partition_start = p * K;
            const size_t partition_end = std::min(partition_start + K, m_original_size);
            const size_t range_start = std::max(startIndex, partition_start);
            const size_t range_end = std::min(endIndex, partition_end);
            std::vector<T> partition_buffer;
            partition_buffer.reserve(range_end - range_start);
            const size_t written = fetch_partition_range(p, range_start, range_end, partition_buffer);
            if (written == 0) {
                continue;
            }
            std::copy_n(partition_buffer.begin(), written, output.begin() + total_written);
            total_written += written;
        }
        
        return total_written;
    }

    T operator[](size_t index) const override {
        if (index >= m_original_size) [[unlikely]] {
             throw std::out_of_range("index out of range in UniformPartitioning");
        }
        
        // Division and modulo are expensive - but necessary here
        // Compiler will optimize to shift+mask if K is power of 2
        const size_t partition_index = index / K;
        const size_t index_in_partition = index % K;
        
        // Direct object access - NO VIRTUAL CALL if Compressor type is final or compiler can devirtualize
        return partition(partition_index)[index_in_partition];
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
        // K is not serialized as it is a template parameter

        const size_t num_partitions = m_partitions.size();
        ofs.write(reinterpret_cast<const char*>(&num_partitions), sizeof(num_partitions));

        for (size_t i = 0; i < num_partitions; ++i) {
            partition(i).serialize(ofs);
        }
    }

    void load(std::ifstream& ifs) override {
        if (!ifs.is_open()) {
            throw std::runtime_error("Input file stream is not open for loading.");
        }
        ifs.read(reinterpret_cast<char*>(&m_original_size), sizeof(m_original_size));
        // K is not read

        if (ifs.fail()) {
            throw std::runtime_error("Failed to read or invalid data from stream during UniformPartitioning load.");
        }

        size_t num_partitions;
        ifs.read(reinterpret_cast<char*>(&num_partitions), sizeof(num_partitions));
        if (ifs.fail()) {
            throw std::runtime_error("Failed to read number of partitions from stream.");
        }

        m_partitions.clear();
        m_partitions.resize(num_partitions);
        for (size_t i = 0; i < num_partitions; ++i) {
            m_partitions[i].emplace();  // Create empty Compressor
            m_partitions[i]->load(ifs);
        }
    }

private:
    // Using std::optional allows O(1) indexed construction without default-constructing Compressors.
    // resize(N) on vector<optional<T>> just creates N empty optionals (trivially cheap).
    // This enables true constant throughput regardless of partition count.
    std::vector<std::optional<Compressor>> m_partitions;
    size_t m_original_size;
    // m_block_size is now K (template parameter)
    
    // Helper to access partition - handles the optional unwrapping
    const Compressor& partition(size_t i) const { return *m_partitions[i]; }
    Compressor& partition(size_t i) { return *m_partitions[i]; }
};

} // namespace gef

#endif // UNIFORM_PARTITIONING_HPP
