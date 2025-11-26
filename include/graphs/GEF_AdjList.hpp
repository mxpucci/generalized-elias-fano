//
// Created by Gemini on 26/07/25.
//

#ifndef GEF_ADJ_LIST_HPP
#define GEF_ADJ_LIST_HPP

#include "../gef/MyEF.hpp"
#include "../gef/IGEF.hpp"
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <type_traits>
#include <fstream>

namespace gef {

    /**
     * @brief A compressed Adjacency List using Elias-Fano based structures.
     * * @tparam GEFType The type of the compressed structure used for the concatenated neighbors array A.
     * Must satisfy the IGEF interface.
     */
    template <typename GEFType>
    class GEF_AdjList {
    public:
        // Deduce the value type stored in GEFType (the node ID type)
        using T = std::decay_t<decltype(std::declval<GEFType>()[0])>;

    private:
        size_t n_;
        size_t m_;
        
        // Offset array O: Stores the starting index in A for each node.
        // Encoded with MyEF<size_t> as offsets are monotonic and can exceed 32-bit.
        MyEF<size_t> offsets_;
        
        // Neighbors array A: Concatenation of sorted neighborhoods.
        // Encoded with the provided GEFType (e.g., B_GEF, MyEF, etc.)
        GEFType neighbors_;

    public:
        // Default constructor
        GEF_AdjList() : n_(0), m_(0) {}

        /**
         * @brief Constructs the compressed adjacency list.
         * * @param edges Vector of pairs representing edges (source, destination).
         * @param n Number of nodes.
         * @param m Number of edges.
         * @param reverse_edges If true, edges are treated as (destination, source).
         * @param bv_factory Factory for creating bit vectors (required by GEF structures).
         * @param metrics Optional metrics collector.
         */
        GEF_AdjList(
            const std::vector<std::pair<size_t, size_t>>& edges,
            size_t n,
            size_t m,
            bool reverse_edges = false,
            std::shared_ptr<IBitVectorFactory> bv_factory = nullptr,
            CompressionBuildMetrics* metrics = nullptr
        ) : n_(n), m_(m) {
            
            // Validate inputs
            if (n == 0 && m > 0) {
                 throw std::invalid_argument("Cannot have edges with 0 nodes");
            }

            // 1. Build temporary adjacency list
            // Using vector of vectors to sort neighbors locally.
            std::vector<std::vector<T>> temp_adj(n);

            for (const auto& edge : edges) {
                size_t u = edge.first;
                size_t v = edge.second;

                if (reverse_edges) {
                    std::swap(u, v);
                }

                // Sanity checks
                if (u >= n || v >= n) {
                    continue; // Or throw, ignoring out of bounds for safety
                }

                temp_adj[u].push_back(static_cast<T>(v));
            }

            // 2. Flatten and create Offset array
            // O has size n + 1. O[i] = start index of neighbors of i.
            std::vector<size_t> offset_values;
            offset_values.reserve(n + 1);
            
            std::vector<T> neighbor_values;
            neighbor_values.reserve(m);

            size_t current_offset = 0;
            offset_values.push_back(current_offset);

            for (size_t i = 0; i < n; ++i) {
                // Ensure neighbors are locally sorted
                std::sort(temp_adj[i].begin(), temp_adj[i].end());
                
                // Remove duplicates
                temp_adj[i].erase(std::unique(temp_adj[i].begin(), temp_adj[i].end()), temp_adj[i].end());
                
                // Append to flat array
                neighbor_values.insert(neighbor_values.end(), temp_adj[i].begin(), temp_adj[i].end());
                
                current_offset += temp_adj[i].size();
                offset_values.push_back(current_offset);
            }

            // Verify m matches computed edges (or update m if filtered)
            if (current_offset != neighbor_values.size()) {
                 throw std::runtime_error("Mismatch in edge counting during construction");
            }
            m_ = current_offset; // Update strictly

            // 3. Compress Offsets (O) using MyEF
            // Use default factory if null provided, though usually passed from caller
            if (!bv_factory) {
                bv_factory = std::make_shared<SDSLBitVectorFactory>(); 
            }

            offsets_ = MyEF<size_t>(bv_factory, offset_values, OPTIMAL_SPLIT_POINT, metrics);

            // 4. Compress Neighbors (A) using GEFType template
            neighbors_ = GEFType(bv_factory, neighbor_values, OPTIMAL_SPLIT_POINT, metrics);
        }

        /**
         * @brief Returns the i-th neighbor of the given node.
         * * @param node_index The node ID.
         * @param i The index within the neighbor list.
         * @return T The ID of the neighbor.
         */
        T getIthNeighbor(size_t node_index, size_t i) const {
            if (node_index >= n_) {
                throw std::out_of_range("Node index out of bounds");
            }

            size_t start = offsets_[node_index];
            size_t end = offsets_[node_index + 1];
            
            if (start + i >= end) {
                throw std::out_of_range("Neighbor index out of bounds");
            }

            return neighbors_[start + i];
        }

        /**
         * @brief Returns all neighbors of the given node.
         * * @param node_index The node ID.
         * @return std::vector<T> List of neighbors.
         */
        std::vector<T> getNeighbors(size_t node_index) const {
            if (node_index >= n_) {
                return {};
            }

            size_t start = offsets_[node_index];
            size_t end = offsets_[node_index + 1];
            size_t count = end - start;

            if (count == 0) {
                return {};
            }

            std::vector<T> res(count);
            neighbors_.get_elements(start, count, res);
            return res;
        }

        /**
         * @brief Returns the degree of a node.
         */
        size_t degree(size_t node_index) const {
            if (node_index >= n_) return 0;
            return offsets_[node_index + 1] - offsets_[node_index];
        }

        /**
         * @brief Serializes the structure to a file stream.
         */
        void serialize(std::ofstream& ofs) const {
            if (!ofs.is_open()) {
                throw std::runtime_error("OutputStream not open");
            }
            ofs.write(reinterpret_cast<const char*>(&n_), sizeof(n_));
            ofs.write(reinterpret_cast<const char*>(&m_), sizeof(m_));
            offsets_.serialize(ofs);
            neighbors_.serialize(ofs);
        }

        /**
         * @brief Loads the structure from a file stream.
         */
        void load(std::ifstream& ifs, std::shared_ptr<IBitVectorFactory> bv_factory) {
            if (!ifs.is_open()) {
                throw std::runtime_error("InputStream not open");
            }
            ifs.read(reinterpret_cast<char*>(&n_), sizeof(n_));
            ifs.read(reinterpret_cast<char*>(&m_), sizeof(m_));
            offsets_.load(ifs, bv_factory);
            neighbors_.load(ifs, bv_factory);
        }

        /**
         * @brief Returns total size in bytes.
         */
        size_t size_in_bytes() const {
            return sizeof(n_) + sizeof(m_) /*+ offsets_.size_in_bytes()*/ + neighbors_.size_in_bytes();
        }

        size_t num_nodes() const { return n_; }
        size_t num_edges() const { return m_; }
    };

} // namespace gef

#endif // GEF_ADJ_LIST_HPP