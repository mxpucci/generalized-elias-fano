//
// Graph compression utility
//
// Takes a SNAP-format graph file path and outputs:
// - size_in_bytes: total compressed size
// - compression_ratio: compressed_size / (2 * #edges * 8)
//

#include "../include/graphs/GEF_Graph.hpp"
#include "../include/graphs/GraphPreprocessor.hpp"
#include "../include/gef/U_GEF.hpp"
#include "../include/gef/UniformedPartitioner.hpp"
#include "../include/datastructures/PastaBitVectorFactory.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace gef;

/**
 * @brief Adapter for U_GEF that reorders constructor arguments to match
 * what UniformedPartitioner expects: (data, args...) instead of (bv_factory, data, strategy).
 */
template<typename T, typename BitVectorType = PastaBitVector>
class U_GEF_Adapter : public U_GEF<T, BitVectorType> {
public:
    using Base = U_GEF<T, BitVectorType>;
    
    U_GEF_Adapter() = default;
    
    // Constructor that accepts (data, bv_factory, strategy) - reorders to (bv_factory, data, strategy)
    template<typename C>
    U_GEF_Adapter(const C& data,
                  const std::shared_ptr<IBitVectorFactory>& bv_factory,
                  SplitPointStrategy strategy)
        : Base(bv_factory, data, strategy) {}
    
    // Also accept Span
    U_GEF_Adapter(const Span<const T>& data,
                  const std::shared_ptr<IBitVectorFactory>& bv_factory,
                  SplitPointStrategy strategy)
        : Base(bv_factory, data, strategy) {}
};

/**
 * @brief Wrapper that adapts UniformedPartitioner<T, U_GEF_Adapter<T>> to the interface
 * expected by GEF_AdjList (bv_factory, data, strategy, metrics).
 */
template<typename T, typename BitVectorType = PastaBitVector>
class PartitionedUGEF : public IGEF<T> {
private:
    static constexpr size_t PARTITION_SIZE = 32768; // 32k
    
    using InnerCompressor = U_GEF_Adapter<T, BitVectorType>;
    using Partitioner = UniformedPartitioner<T, InnerCompressor, 
                                             std::shared_ptr<IBitVectorFactory>, 
                                             SplitPointStrategy>;
    
    std::unique_ptr<Partitioner> partitioner_;

public:
    using IGEF<T>::serialize;
    using IGEF<T>::load;

    PartitionedUGEF() = default;
    
    template<typename C>
    PartitionedUGEF(const std::shared_ptr<IBitVectorFactory>& bv_factory,
                    const C& data,
                    SplitPointStrategy strategy = OPTIMAL_SPLIT_POINT,
                    CompressionBuildMetrics* /*metrics*/ = nullptr) {
        // Convert to vector if needed
        std::vector<T> vec(data.begin(), data.end());
        partitioner_ = std::make_unique<Partitioner>(vec, PARTITION_SIZE, bv_factory, strategy);
    }
    
    // Copy constructor
    PartitionedUGEF(const PartitionedUGEF& other) = delete;
    PartitionedUGEF& operator=(const PartitionedUGEF& other) = delete;
    
    // Move constructor
    PartitionedUGEF(PartitionedUGEF&& other) noexcept = default;
    PartitionedUGEF& operator=(PartitionedUGEF&& other) noexcept = default;
    
    ~PartitionedUGEF() override = default;

    T operator[](size_t index) const override {
        return (*partitioner_)[index];
    }
    
    size_t get_elements(size_t startIndex, size_t count, std::vector<T>& output) const override {
        return partitioner_->get_elements(startIndex, count, output);
    }

    size_t size() const override {
        return partitioner_ ? partitioner_->size() : 0;
    }

    size_t size_in_bytes() const override {
        return partitioner_ ? partitioner_->size_in_bytes() : 0;
    }

    size_t theoretical_size_in_bytes() const override {
        return partitioner_ ? partitioner_->theoretical_size_in_bytes() : 0;
    }

    uint8_t split_point() const override {
        return 0; // Partitioner doesn't have a single split point
    }

    void serialize(std::ofstream& ofs) const override {
        if (partitioner_) {
            partitioner_->serialize(ofs);
        }
    }

    void load(std::ifstream& ifs, const std::shared_ptr<IBitVectorFactory> bv_factory) override {
        partitioner_ = std::make_unique<Partitioner>();
        partitioner_->load(ifs, bv_factory);
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file.txt>" << std::endl;
        std::cerr << "\nCompresses a SNAP-format graph and outputs compression statistics." << std::endl;
        std::cerr << "\nUses UniformedPartitioner with U_GEF (OPTIMAL strategy, 32k partitions)" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    
    std::cout << "Loading graph from: " << filepath << std::endl;
    std::cout << "Compressor: UniformedPartitioner<U_GEF> with OPTIMAL_SPLIT_POINT" << std::endl;
    std::cout << "Partition size: 32k" << std::endl;

    // Create bit vector factory
    auto bv_factory = std::make_shared<PastaBitVectorFactory>();

    // Time the loading process
    auto start = std::chrono::steady_clock::now();

    // Load and compress the graph using PartitionedUGEF
    GEF_Graph<PartitionedUGEF<uint32_t, PastaBitVector>> graph(filepath, bv_factory);

    auto end = std::chrono::steady_clock::now();
    double build_time = std::chrono::duration<double>(end - start).count();

    // Output statistics
    std::cout << std::endl;
    std::cout << "=== Graph Statistics ===" << std::endl;
    std::cout << "Nodes: " << graph.num_nodes() << std::endl;
    std::cout << "Edges: " << graph.num_edges() << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Compression Statistics ===" << std::endl;
    std::cout << "size_in_bytes: " << graph.size_in_bytes() << std::endl;
    std::cout << "compression_ratio: " << std::fixed << std::setprecision(6) 
              << graph.compression_ratio() << std::endl;
    std::cout << "bits_per_edge: " << std::fixed << std::setprecision(2) 
              << graph.bits_per_edge() << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Build Time ===" << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(3) 
              << build_time << " seconds" << std::endl;

    return 0;
}
