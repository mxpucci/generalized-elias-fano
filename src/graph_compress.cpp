//
// Graph compression utility
//
// Takes a SNAP-format graph file path and outputs:
// - size_in_bytes: total compressed size
// - compression_ratio: compressed_size / (2 * #edges * 8)
//
// Modes:
// - ugef-rcm (default): U_GEF with Reverse Cuthill-McKee node ordering
// - bgef-spectral: B_GEF_STAR with spectral (Fiedler) node ordering
// - rlegef-rcm: RLE_GEF with Reverse Cuthill-McKee node ordering
//

#include "../include/graphs/GEF_Graph.hpp"
#include "../include/graphs/GraphPreprocessor.hpp"
#include "../include/graphs/U_GEF_AdjList.hpp"
#include "../include/gef/U_GEF.hpp"
#include "../include/gef/B_GEF.hpp"
#include "../include/gef/B_GEF_STAR.hpp"
#include "../include/gef/RLE_GEF.hpp"
#include "../include/gef/UniformedPartitioner.hpp"
#include "../include/datastructures/PastaBitVectorFactory.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstring>

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

/**
 * @brief Adapter for B_GEF_STAR that reorders constructor arguments to match
 * what UniformedPartitioner expects: (data, args...) instead of (bv_factory, data, strategy).
 */
template<typename T, typename BitVectorType = PastaBitVector>
class B_GEF_STAR_Adapter : public B_GEF_STAR<T, BitVectorType> {
public:
    using Base = B_GEF_STAR<T, BitVectorType>;
    
    B_GEF_STAR_Adapter() = default;
    
    // Constructor that accepts (data, bv_factory, strategy) - reorders to (bv_factory, data, strategy)
    template<typename C>
    B_GEF_STAR_Adapter(const C& data,
                  const std::shared_ptr<IBitVectorFactory>& bv_factory,
                  SplitPointStrategy strategy)
        : Base(bv_factory, data, strategy) {}
    
    // Also accept Span
    B_GEF_STAR_Adapter(const Span<const T>& data,
                  const std::shared_ptr<IBitVectorFactory>& bv_factory,
                  SplitPointStrategy strategy)
        : Base(bv_factory, data, strategy) {}
};

/**
 * @brief Wrapper that adapts UniformedPartitioner<T, B_GEF_STAR_Adapter<T>> to the interface
 * expected by GEF_AdjList (bv_factory, data, strategy, metrics).
 */
template<typename T, typename BitVectorType = PastaBitVector>
class PartitionedBGEF : public IGEF<T> {
private:
    static constexpr size_t PARTITION_SIZE = 1024; // 8k
    
    using InnerCompressor = B_GEF_STAR_Adapter<T, BitVectorType>;
    using Partitioner = UniformedPartitioner<T, InnerCompressor, 
                                             std::shared_ptr<IBitVectorFactory>, 
                                             SplitPointStrategy>;
    
    std::unique_ptr<Partitioner> partitioner_;

public:
    using IGEF<T>::serialize;
    using IGEF<T>::load;

    PartitionedBGEF() = default;
    
    template<typename C>
    PartitionedBGEF(const std::shared_ptr<IBitVectorFactory>& bv_factory,
                    const C& data,
                    SplitPointStrategy strategy = OPTIMAL_SPLIT_POINT,
                    CompressionBuildMetrics* /*metrics*/ = nullptr) {
        // Convert to vector if needed
        std::vector<T> vec(data.begin(), data.end());
        partitioner_ = std::make_unique<Partitioner>(vec, PARTITION_SIZE, bv_factory, strategy);
    }
    
    // Copy constructor
    PartitionedBGEF(const PartitionedBGEF& other) = delete;
    PartitionedBGEF& operator=(const PartitionedBGEF& other) = delete;
    
    // Move constructor
    PartitionedBGEF(PartitionedBGEF&& other) noexcept = default;
    PartitionedBGEF& operator=(PartitionedBGEF&& other) noexcept = default;
    
    ~PartitionedBGEF() override = default;

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

/**
 * @brief Adapter for RLE_GEF that reorders constructor arguments to match
 * what UniformedPartitioner expects: (data, args...) instead of (bv_factory, data).
 */
template<typename T, typename BitVectorType = PastaBitVector>
class RLE_GEF_Adapter : public RLE_GEF<T, BitVectorType> {
public:
    using Base = RLE_GEF<T, BitVectorType>;
    
    RLE_GEF_Adapter() = default;
    
    // Constructor that accepts (data, bv_factory) - reorders to (bv_factory, data)
    template<typename C>
    RLE_GEF_Adapter(const C& data,
                    const std::shared_ptr<IBitVectorFactory>& bv_factory)
        : Base(bv_factory, data) {}
    
    // Also accept Span
    RLE_GEF_Adapter(const Span<const T>& data,
                    const std::shared_ptr<IBitVectorFactory>& bv_factory)
        : Base(bv_factory, data) {}
};

/**
 * @brief Wrapper that adapts UniformedPartitioner<T, RLE_GEF_Adapter<T>> to the interface
 * expected by GEF_AdjList (bv_factory, data, strategy, metrics).
 */
template<typename T, typename BitVectorType = PastaBitVector>
class PartitionedRLEGEF : public IGEF<T> {
private:
    static constexpr size_t PARTITION_SIZE = 32768; // 32k
    
    using InnerCompressor = RLE_GEF_Adapter<T, BitVectorType>;
    using Partitioner = UniformedPartitioner<T, InnerCompressor, 
                                             std::shared_ptr<IBitVectorFactory>>;
    
    std::unique_ptr<Partitioner> partitioner_;

public:
    using IGEF<T>::serialize;
    using IGEF<T>::load;

    PartitionedRLEGEF() = default;
    
    template<typename C>
    PartitionedRLEGEF(const std::shared_ptr<IBitVectorFactory>& bv_factory,
                      const C& data,
                      SplitPointStrategy /*strategy*/ = OPTIMAL_SPLIT_POINT,
                      CompressionBuildMetrics* /*metrics*/ = nullptr) {
        // Convert to vector if needed
        std::vector<T> vec(data.begin(), data.end());
        partitioner_ = std::make_unique<Partitioner>(vec, PARTITION_SIZE, bv_factory);
    }
    
    // Copy constructor
    PartitionedRLEGEF(const PartitionedRLEGEF& other) = delete;
    PartitionedRLEGEF& operator=(const PartitionedRLEGEF& other) = delete;
    
    // Move constructor
    PartitionedRLEGEF(PartitionedRLEGEF&& other) noexcept = default;
    PartitionedRLEGEF& operator=(PartitionedRLEGEF&& other) noexcept = default;
    
    ~PartitionedRLEGEF() override = default;

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

/**
 * @brief Runs compression with U_GEF and Reverse Cuthill-McKee ordering (default mode).
 */
void run_ugef_rcm(const std::string& filepath, std::shared_ptr<IBitVectorFactory> bv_factory) {
    std::cout << "Compressor: UniformedPartitioner<U_GEF> with OPTIMAL_SPLIT_POINT" << std::endl;
    std::cout << "Node ordering: Reverse Cuthill-McKee (RCM)" << std::endl;
    std::cout << "Partition size: 32k" << std::endl;

    // Time the loading process
    auto start = std::chrono::steady_clock::now();

    // Parse metadata to get expected node count
    auto [expected_nodes, expected_edges] = GraphPreprocessor::parse_metadata(filepath);
    
    // Preprocess with RCM ordering
    std::cout << "\nComputing RCM ordering..." << std::endl;
    auto rcm_start = std::chrono::steady_clock::now();
    
    PreprocessedGraph preprocessed = (expected_nodes > 0) 
        ? GraphPreprocessor::preprocess_rcm(filepath, expected_nodes)
        : GraphPreprocessor::preprocess_rcm(filepath);
    
    auto rcm_end = std::chrono::steady_clock::now();
    double rcm_time = std::chrono::duration<double>(rcm_end - rcm_start).count();
    std::cout << "RCM ordering time: " << std::fixed << std::setprecision(3) 
              << rcm_time << " seconds" << std::endl;

    // Build the graph with U_GEF
    GEF_Graph<PartitionedUGEF<uint32_t, PastaBitVector>> graph(preprocessed, bv_factory);

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
    std::cout << "Total time: " << std::fixed << std::setprecision(3) 
              << build_time << " seconds" << std::endl;
}

/**
 * @brief Runs compression with B_GEF_STAR and spectral (Fiedler) ordering.
 */
void run_bgef_spectral(const std::string& filepath, std::shared_ptr<IBitVectorFactory> bv_factory) {
    std::cout << "Compressor: UniformedPartitioner<B_GEF_STAR> with OPTIMAL_SPLIT_POINT" << std::endl;
    std::cout << "Node ordering: Spectral (Fiedler vector)" << std::endl;
    std::cout << "Partition size: 8k" << std::endl;

    // Time the loading process
    auto start = std::chrono::steady_clock::now();

    // Parse metadata to get expected node count
    auto [expected_nodes, expected_edges] = GraphPreprocessor::parse_metadata(filepath);
    
    // Preprocess with spectral ordering
    std::cout << "\nComputing spectral ordering..." << std::endl;
    auto spectral_start = std::chrono::steady_clock::now();
    
    PreprocessedGraph preprocessed = (expected_nodes > 0) 
        ? GraphPreprocessor::preprocess_spectral(filepath, expected_nodes)
        : GraphPreprocessor::preprocess_spectral(filepath);
    
    auto spectral_end = std::chrono::steady_clock::now();
    double spectral_time = std::chrono::duration<double>(spectral_end - spectral_start).count();
    std::cout << "Spectral ordering time: " << std::fixed << std::setprecision(3) 
              << spectral_time << " seconds" << std::endl;

    // Build the graph with B_GEF_STAR
    GEF_Graph<PartitionedBGEF<uint32_t, PastaBitVector>> graph(preprocessed, bv_factory);

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
    std::cout << "Total time: " << std::fixed << std::setprecision(3) 
              << build_time << " seconds" << std::endl;
}

/**
 * @brief Runs compression with RLE_GEF and Reverse Cuthill-McKee ordering.
 */
void run_rlegef_rcm(const std::string& filepath, std::shared_ptr<IBitVectorFactory> bv_factory) {
    std::cout << "Compressor: UniformedPartitioner<RLE_GEF>" << std::endl;
    std::cout << "Node ordering: Reverse Cuthill-McKee (RCM)" << std::endl;
    std::cout << "Partition size: 32k" << std::endl;

    // Time the loading process
    auto start = std::chrono::steady_clock::now();

    // Parse metadata to get expected node count
    auto [expected_nodes, expected_edges] = GraphPreprocessor::parse_metadata(filepath);
    
    // Preprocess with RCM ordering
    std::cout << "\nComputing RCM ordering..." << std::endl;
    auto rcm_start = std::chrono::steady_clock::now();
    
    PreprocessedGraph preprocessed = (expected_nodes > 0) 
        ? GraphPreprocessor::preprocess_rcm(filepath, expected_nodes)
        : GraphPreprocessor::preprocess_rcm(filepath);
    
    auto rcm_end = std::chrono::steady_clock::now();
    double rcm_time = std::chrono::duration<double>(rcm_end - rcm_start).count();
    std::cout << "RCM ordering time: " << std::fixed << std::setprecision(3) 
              << rcm_time << " seconds" << std::endl;

    // Build the graph with RLE_GEF
    GEF_Graph<PartitionedRLEGEF<uint32_t, PastaBitVector>> graph(preprocessed, bv_factory);

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
    std::cout << "Total time: " << std::fixed << std::setprecision(3) 
              << build_time << " seconds" << std::endl;
}

/**
 * @brief Runs compression with U_GEF_AdjList and Reverse Cuthill-McKee ordering.
 * 
 * This mode uses the unified U_GEF_AdjList structure that combines offset
 * encoding (via MyEF/Elias-Fano) with gap-encoded neighbors. Exceptions are
 * forced at neighborhood boundaries, allowing the offset array to serve as
 * the exception bitvector.
 */
void run_ugef_adjlist_rcm(const std::string& filepath, std::shared_ptr<IBitVectorFactory> bv_factory) {
    std::cout << "Compressor: U_GEF_AdjList (unified offset + neighbor encoding)" << std::endl;
    std::cout << "Node ordering: Reverse Cuthill-McKee (RCM)" << std::endl;

    // Time the loading process
    auto start = std::chrono::steady_clock::now();

    // Parse metadata to get expected node count
    auto [expected_nodes, expected_edges] = GraphPreprocessor::parse_metadata(filepath);
    
    // Preprocess with RCM ordering
    std::cout << "\nComputing RCM ordering..." << std::endl;
    auto rcm_start = std::chrono::steady_clock::now();
    
    PreprocessedGraph preprocessed = (expected_nodes > 0) 
        ? GraphPreprocessor::preprocess_rcm(filepath, expected_nodes)
        : GraphPreprocessor::preprocess_rcm(filepath);
    
    auto rcm_end = std::chrono::steady_clock::now();
    double rcm_time = std::chrono::duration<double>(rcm_end - rcm_start).count();
    std::cout << "RCM ordering time: " << std::fixed << std::setprecision(3) 
              << rcm_time << " seconds" << std::endl;

    const size_t n = preprocessed.num_nodes;
    const size_t m = preprocessed.num_edges;

    // Build out-adjacency list (forward edges) using move semantics
    std::cout << "\nBuilding out-adjacency list..." << std::endl;
    auto out_start = std::chrono::steady_clock::now();
    U_GEF_AdjList<uint32_t, PastaBitVector> out_adj(
        std::move(preprocessed.edges), n, false, bv_factory
    );
    auto out_end = std::chrono::steady_clock::now();
    double out_time = std::chrono::duration<double>(out_end - out_start).count();
    std::cout << "Out-adjacency build time: " << std::fixed << std::setprecision(3) 
              << out_time << " seconds" << std::endl;

    // For in-adjacency, we need to reconstruct edges from out_adj
    // This is more memory efficient than keeping two copies of edges
    std::cout << "Building in-adjacency list..." << std::endl;
    auto in_start = std::chrono::steady_clock::now();
    
    // Reconstruct edges from out_adj (memory efficient: build incrementally)
    std::vector<std::pair<size_t, size_t>> reversed_edges;
    reversed_edges.reserve(out_adj.num_edges());
    for (size_t src = 0; src < n; ++src) {
        auto neighbors = out_adj.getNeighbors(src);
        for (const auto& dst : neighbors) {
            reversed_edges.emplace_back(dst, src);  // Reversed: dst -> src
        }
    }
    
    U_GEF_AdjList<uint32_t, PastaBitVector> in_adj(
        std::move(reversed_edges), n, false, bv_factory  // Already reversed
    );
    auto in_end = std::chrono::steady_clock::now();
    double in_time = std::chrono::duration<double>(in_end - in_start).count();
    std::cout << "In-adjacency build time: " << std::fixed << std::setprecision(3) 
              << in_time << " seconds" << std::endl;

    auto end = std::chrono::steady_clock::now();
    double build_time = std::chrono::duration<double>(end - start).count();

    // Compute total size
    size_t total_size = out_adj.size_in_bytes() + in_adj.size_in_bytes();
    
    // Compression ratio: compressed_size / (2 * #edges * 8 bytes per edge)
    // Each edge is stored as two 64-bit integers (source, dest) in uncompressed form
    double raw_size = 2.0 * static_cast<double>(m) * sizeof(uint64_t);
    double compression_ratio = static_cast<double>(total_size) / raw_size;
    double bits_per_edge = (static_cast<double>(total_size) * 8.0) / static_cast<double>(m);

    // Output statistics
    std::cout << std::endl;
    std::cout << "=== Graph Statistics ===" << std::endl;
    std::cout << "Nodes: " << n << std::endl;
    std::cout << "Edges: " << m << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Compression Statistics ===" << std::endl;
    std::cout << "Out-adjacency size: " << out_adj.size_in_bytes() << " bytes" << std::endl;
    std::cout << "In-adjacency size: " << in_adj.size_in_bytes() << " bytes" << std::endl;
    std::cout << "size_in_bytes (total): " << total_size << std::endl;
    std::cout << "compression_ratio: " << std::fixed << std::setprecision(6) 
              << compression_ratio << std::endl;
    std::cout << "bits_per_edge: " << std::fixed << std::setprecision(2) 
              << bits_per_edge << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Build Time ===" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(3) 
              << build_time << " seconds" << std::endl;
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <graph_file.txt> [mode]" << std::endl;
    std::cerr << "\nCompresses a SNAP-format graph and outputs compression statistics." << std::endl;
    std::cerr << "\nModes:" << std::endl;
    std::cerr << "  ugef-rcm         (default) U_GEF with Reverse Cuthill-McKee node ordering" << std::endl;
    std::cerr << "  ugef-adjlist-rcm           U_GEF_AdjList (unified structure) with RCM ordering" << std::endl;
    std::cerr << "  bgef-spectral              B_GEF_STAR with spectral (Fiedler) node ordering" << std::endl;
    std::cerr << "  rlegef-rcm                 RLE_GEF with Reverse Cuthill-McKee node ordering" << std::endl;
    std::cerr << "\nExamples:" << std::endl;
    std::cerr << "  " << program_name << " graph.txt" << std::endl;
    std::cerr << "  " << program_name << " graph.txt ugef-rcm" << std::endl;
    std::cerr << "  " << program_name << " graph.txt ugef-adjlist-rcm" << std::endl;
    std::cerr << "  " << program_name << " graph.txt bgef-spectral" << std::endl;
    std::cerr << "  " << program_name << " graph.txt rlegef-rcm" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string filepath = argv[1];
    std::string mode = "ugef-rcm";  // Default mode
    
    if (argc >= 3) {
        mode = argv[2];
    }

    // Validate mode
    if (mode != "ugef-rcm" && mode != "ugef-adjlist-rcm" && mode != "bgef-spectral" && mode != "rlegef-rcm") {
        std::cerr << "Error: Unknown mode '" << mode << "'" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Loading graph from: " << filepath << std::endl;
    
    // Create bit vector factory
    auto bv_factory = std::make_shared<PastaBitVectorFactory>();

    if (mode == "ugef-rcm") {
        run_ugef_rcm(filepath, bv_factory);
    } else if (mode == "ugef-adjlist-rcm") {
        run_ugef_adjlist_rcm(filepath, bv_factory);
    } else if (mode == "bgef-spectral") {
        run_bgef_spectral(filepath, bv_factory);
    } else if (mode == "rlegef-rcm") {
        run_rlegef_rcm(filepath, bv_factory);
    }

    return 0;
}
