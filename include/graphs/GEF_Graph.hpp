//
// Created by Michelangelo Pucci on 26/11/25.
//

#ifndef GEF_GRAPH_HPP
#define GEF_GRAPH_HPP

#include "GEF_AdjList.hpp"
#include "GraphPreprocessor.hpp"
#include "../gef/B_GEF.hpp"
#include "../gef/MyEF.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"
#include "../datastructures/PastaBitVector.hpp"
#include <memory>
#include <fstream>
#include <filesystem>

namespace gef {

    /**
     * @brief A compressed directed graph representation using GEF-based adjacency lists.
     * 
     * Stores both the forward (outgoing) and reverse (incoming) adjacency lists
     * using Elias-Fano based compression structures. Nodes are expected to be
     * relabeled according to BFS order before construction for better compression.
     * 
     * @tparam GEFType The GEF variant to use for neighbor arrays (e.g., B_GEF, MyEF)
     */
    template <typename GEFType = B_GEF<uint32_t, PastaBitVector>>
    class GEF_Graph {
    public:
        using T = typename GEF_AdjList<GEFType>::T;

    private:
        // Forward adjacency list: out-neighbors of each node
        GEF_AdjList<GEFType> forward_adj_;
        
        // Reverse adjacency list: in-neighbors of each node
        GEF_AdjList<GEFType> reverse_adj_;

        // Node ID mappings (optional, for translating back to original IDs)
        std::vector<size_t> new_to_original_;
        
        // Graph metadata
        size_t n_;  // Number of nodes
        size_t m_;  // Number of edges

    public:
        // Default constructor
        GEF_Graph() : n_(0), m_(0) {}

        /**
         * @brief Constructs a GEF_Graph from preprocessed graph data.
         * 
         * @param preprocessed The preprocessed graph with BFS-relabeled edges
         * @param bv_factory Bit vector factory for GEF structures
         * @param metrics Optional metrics collector
         */
        GEF_Graph(
            const PreprocessedGraph& preprocessed,
            std::shared_ptr<IBitVectorFactory> bv_factory = nullptr,
            CompressionBuildMetrics* metrics = nullptr
        ) : n_(preprocessed.num_nodes), 
            m_(preprocessed.num_edges),
            new_to_original_(preprocessed.new_to_original) {
            
            if (!bv_factory) {
                bv_factory = std::make_shared<SDSLBitVectorFactory>();
            }

            // Build forward adjacency (edges as-is)
            forward_adj_ = GEF_AdjList<GEFType>(
                preprocessed.edges,
                n_,
                m_,
                false,  // Do not reverse edges
                bv_factory,
                metrics
            );

            // Build reverse adjacency (reverse the edges)
            reverse_adj_ = GEF_AdjList<GEFType>(
                preprocessed.edges,
                n_,
                m_,
                true,   // Reverse edges: (u,v) becomes (v,u)
                bv_factory,
                metrics
            );
        }

        /**
         * @brief Constructs a GEF_Graph directly from a SNAP file.
         * 
         * Convenience constructor that handles preprocessing internally.
         * 
         * @param filepath Path to the SNAP-format edge list file
         * @param bv_factory Bit vector factory for GEF structures
         * @param metrics Optional metrics collector
         */
        GEF_Graph(
            const std::filesystem::path& filepath,
            std::shared_ptr<IBitVectorFactory> bv_factory = nullptr,
            CompressionBuildMetrics* metrics = nullptr
        ) {
            // Parse metadata to get expected node count
            auto [expected_nodes, expected_edges] = GraphPreprocessor::parse_metadata(filepath);
            
            // Preprocess the graph
            PreprocessedGraph preprocessed = (expected_nodes > 0) 
                ? GraphPreprocessor::preprocess(filepath, expected_nodes)
                : GraphPreprocessor::preprocess(filepath);
            
            n_ = preprocessed.num_nodes;
            m_ = preprocessed.num_edges;
            new_to_original_ = std::move(preprocessed.new_to_original);

            if (!bv_factory) {
                bv_factory = std::make_shared<SDSLBitVectorFactory>();
            }

            // Build forward adjacency
            forward_adj_ = GEF_AdjList<GEFType>(
                preprocessed.edges,
                n_,
                m_,
                false,
                bv_factory,
                metrics
            );

            // Build reverse adjacency
            reverse_adj_ = GEF_AdjList<GEFType>(
                preprocessed.edges,
                n_,
                m_,
                true,
                bv_factory,
                metrics
            );
        }

        /**
         * @brief Constructs a GEF_Graph from raw edges (already relabeled).
         * 
         * @param edges Vector of (source, dest) pairs
         * @param num_nodes Number of nodes
         * @param bv_factory Bit vector factory
         * @param metrics Optional metrics collector
         */
        GEF_Graph(
            const std::vector<std::pair<size_t, size_t>>& edges,
            size_t num_nodes,
            std::shared_ptr<IBitVectorFactory> bv_factory = nullptr,
            CompressionBuildMetrics* metrics = nullptr
        ) : n_(num_nodes), m_(edges.size()) {
            
            if (!bv_factory) {
                bv_factory = std::make_shared<SDSLBitVectorFactory>();
            }

            forward_adj_ = GEF_AdjList<GEFType>(edges, n_, m_, false, bv_factory, metrics);
            reverse_adj_ = GEF_AdjList<GEFType>(edges, n_, m_, true, bv_factory, metrics);
        }

        // ===== Forward Graph Queries =====

        /**
         * @brief Returns the i-th out-neighbor of a node.
         */
        T out_neighbor(size_t node, size_t i) const {
            return forward_adj_.getIthNeighbor(node, i);
        }

        /**
         * @brief Returns all out-neighbors of a node.
         */
        std::vector<T> out_neighbors(size_t node) const {
            return forward_adj_.getNeighbors(node);
        }

        /**
         * @brief Returns the out-degree of a node.
         */
        size_t out_degree(size_t node) const {
            return forward_adj_.degree(node);
        }

        // ===== Reverse Graph Queries =====

        /**
         * @brief Returns the i-th in-neighbor of a node.
         */
        T in_neighbor(size_t node, size_t i) const {
            return reverse_adj_.getIthNeighbor(node, i);
        }

        /**
         * @brief Returns all in-neighbors of a node.
         */
        std::vector<T> in_neighbors(size_t node) const {
            return reverse_adj_.getNeighbors(node);
        }

        /**
         * @brief Returns the in-degree of a node.
         */
        size_t in_degree(size_t node) const {
            return reverse_adj_.degree(node);
        }

        // ===== Node ID Translation =====

        /**
         * @brief Translates a BFS-ordered node ID back to its original ID.
         * 
         * @param new_id The BFS-ordered node ID
         * @return Original node ID, or SIZE_MAX if mapping not available
         */
        size_t to_original_id(size_t new_id) const {
            if (new_to_original_.empty() || new_id >= new_to_original_.size()) {
                return SIZE_MAX;
            }
            return new_to_original_[new_id];
        }

        /**
         * @brief Checks if original ID mappings are available.
         */
        bool has_original_mapping() const {
            return !new_to_original_.empty();
        }

        // ===== Graph Metadata =====

        size_t num_nodes() const { return n_; }
        size_t num_edges() const { return m_; }

        /**
         * @brief Returns total size in bytes of the compressed structure.
         */
        size_t size_in_bytes() const {
            size_t total = sizeof(n_) + sizeof(m_);
            total += forward_adj_.size_in_bytes();
            total += reverse_adj_.size_in_bytes();
            return total;
        }

        /**
         * @brief Returns compression ratio (compressed size / original size).
         * 
         * Original size assumes 2 * 8 bytes per edge (two 64-bit node IDs).
         * Result < 1.0 means compression achieved.
         */
        double compression_ratio() const {
            size_t original_bytes = m_ * 2 * 8;  // Each edge: 2 node IDs (8 bytes each)
            if (original_bytes == 0) return 0.0;
            return static_cast<double>(size_in_bytes()) / static_cast<double>(original_bytes);
        }

        /**
         * @brief Returns bits per edge for the compressed structure.
         */
        double bits_per_edge() const {
            return static_cast<double>(size_in_bytes() * 8) / static_cast<double>(m_);
        }

        // ===== Accessors for underlying structures =====

        const GEF_AdjList<GEFType>& forward() const { return forward_adj_; }
        const GEF_AdjList<GEFType>& reverse() const { return reverse_adj_; }

        // ===== Serialization =====

        /**
         * @brief Serializes the graph to a file.
         */
        void serialize(std::ofstream& ofs) const {
            if (!ofs.is_open()) {
                throw std::runtime_error("Output stream not open");
            }

            // Write metadata
            ofs.write(reinterpret_cast<const char*>(&n_), sizeof(n_));
            ofs.write(reinterpret_cast<const char*>(&m_), sizeof(m_));

            // Write mappings
            size_t mapping_size = new_to_original_.size();
            ofs.write(reinterpret_cast<const char*>(&mapping_size), sizeof(mapping_size));
            if (mapping_size > 0) {
                ofs.write(reinterpret_cast<const char*>(new_to_original_.data()), 
                          mapping_size * sizeof(size_t));
            }

            // Write adjacency lists
            forward_adj_.serialize(ofs);
            reverse_adj_.serialize(ofs);
        }

        void serialize(const std::filesystem::path& filepath) const {
            std::ofstream ofs(filepath, std::ios::binary);
            if (!ofs.is_open()) {
                throw std::runtime_error("Failed to open file: " + filepath.string());
            }
            serialize(ofs);
        }

        /**
         * @brief Loads the graph from a file.
         */
        void load(std::ifstream& ifs, std::shared_ptr<IBitVectorFactory> bv_factory) {
            if (!ifs.is_open()) {
                throw std::runtime_error("Input stream not open");
            }

            // Read metadata
            ifs.read(reinterpret_cast<char*>(&n_), sizeof(n_));
            ifs.read(reinterpret_cast<char*>(&m_), sizeof(m_));

            // Read mappings
            size_t mapping_size;
            ifs.read(reinterpret_cast<char*>(&mapping_size), sizeof(mapping_size));
            if (mapping_size > 0) {
                new_to_original_.resize(mapping_size);
                ifs.read(reinterpret_cast<char*>(new_to_original_.data()), 
                         mapping_size * sizeof(size_t));
            } else {
                new_to_original_.clear();
            }

            // Read adjacency lists
            forward_adj_.load(ifs, bv_factory);
            reverse_adj_.load(ifs, bv_factory);
        }

        void load(const std::filesystem::path& filepath, 
                  std::shared_ptr<IBitVectorFactory> bv_factory = nullptr) {
            if (!bv_factory) {
                bv_factory = std::make_shared<SDSLBitVectorFactory>();
            }
            std::ifstream ifs(filepath, std::ios::binary);
            if (!ifs.is_open()) {
                throw std::runtime_error("Failed to open file: " + filepath.string());
            }
            load(ifs, bv_factory);
        }
    };

    // Type aliases for common configurations
    using GEF_Graph_BGEF = GEF_Graph<B_GEF<uint32_t, PastaBitVector>>;
    using GEF_Graph_MyEF = GEF_Graph<MyEF<uint32_t, PastaBitVector>>;

} // namespace gef

#endif // GEF_GRAPH_HPP

