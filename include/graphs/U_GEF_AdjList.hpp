//
// Created by Cursor on 29/11/25.
//

#ifndef U_GEF_ADJ_LIST_HPP
#define U_GEF_ADJ_LIST_HPP

#include "../gef/MyEF.hpp"
#include "../gef/IGEF.hpp"
#include "../gef/FastBitWriter.hpp"
#include "../gef/FastUnaryDecoder.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"
#include "../datastructures/PastaBitVector.hpp"
#include "sdsl/int_vector.hpp"

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <type_traits>
#include <fstream>
#include <cmath>

namespace gef {

    /**
     * @brief A compressed Adjacency List using a unified U_GEF-like structure.
     * 
     * This structure exploits the sorted nature of adjacency lists:
     * - The first element of each neighborhood is marked as an "exception"
     * - Subsequent elements within a neighborhood use gap encoding
     * - The offset array (B) is encoded using Elias-Fano (MyEF) for space efficiency
     * 
     * This unifies the offset and neighbor encoding into a single cohesive structure
     * where the offset positions naturally correspond to exception positions.
     * 
     * @tparam T The node ID type (must be integral)
     * @tparam BitVectorType The bit vector implementation for gap encoding
     */
    template <typename T, typename BitVectorType = PastaBitVector>
    class U_GEF_AdjList {
        static_assert(std::is_integral_v<T>, "T must be an integral type");

    public:
        using value_type = T;
        using edge_type = std::pair<size_t, size_t>;

    private:
        size_t n_;  // Number of nodes
        size_t m_;  // Number of edges

        // Base value (minimum neighbor ID across all edges)
        T base_;
        
        // Split point parameters
        uint8_t b_;  // Low bits
        uint8_t h_;  // High bits = total_bits - b_

        /**
         * Offsets encoded with MyEF (replaces B from U_GEF).
         * Stores n+1 values: offsets_[i] = start position of node i's neighbors.
         * offsets_[n] = m (total edges).
         * 
         * This serves the role of the exception bitvector B:
         * - Exception positions are exactly at offsets_[0], offsets_[1], ..., offsets_[n-1]
         *   for nodes with non-empty neighborhoods.
         * - Using MyEF (Elias-Fano) is efficient since offsets are monotonically increasing.
         */
        MyEF<size_t, BitVectorType> offsets_;

        /**
         * Gap vector: unary-encoded gaps for non-exception elements.
         * For each non-first element within a neighborhood, we store the gap
         * between consecutive high parts.
         * 
         * Total size = sum of (gaps + terminators) for all non-first elements.
         */
        std::unique_ptr<BitVectorType> G_;

        /**
         * High parts of exception elements (first neighbor of each non-empty node).
         * Size = number of non-empty neighborhoods.
         */
        sdsl::int_vector<> H_;

        /**
         * Low parts of all neighbors.
         * Size = m (total edges), b_ bits each.
         */
        sdsl::int_vector<> L_;

        // Number of non-empty neighborhoods (= H_.size())
        size_t non_empty_count_;

        static uint8_t bits_for_range(const T min_val, const T max_val) {
            using WI = __int128;
            using WU = unsigned __int128;
            const WI min_w = static_cast<WI>(min_val);
            const WI max_w = static_cast<WI>(max_val);
            const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);
            if (range <= 1) return 1;
            size_t bits = 0;
            WU x = range - 1;
            while (x > 0) { ++bits; x >>= 1; }
            return static_cast<uint8_t>(std::min<size_t>(bits, sizeof(T) * 8));
        }

        // Compute optimal split point for the neighbor array
        static uint8_t compute_split_point(size_t m, T min_val, T max_val) {
            if (m == 0) return 0;
            using WU = unsigned __int128;
            WU u = (static_cast<WU>(max_val) - static_cast<WU>(min_val)) + 1;
            double val = static_cast<double>(u) / static_cast<double>(m);
            if (val <= 1.0) return 0;
            return static_cast<uint8_t>(std::floor(std::log2(val)));
        }

    public:
        // Default constructor
        U_GEF_AdjList() : n_(0), m_(0), base_(0), b_(0), h_(0), non_empty_count_(0) {}

        /**
         * @brief Constructs the compressed adjacency list.
         * 
         * Ultra memory-efficient construction that avoids duplicating edge storage:
         * 1. Works with a mutable copy of edges sorted by source
         * 2. Processes edges in streaming fashion without temp_adj
         * 3. Minimizes peak memory to ~1x edge storage + final structures
         * 
         * @param edges Vector of pairs representing edges (source, destination).
         * @param n Number of nodes.
         * @param m Number of edges (used as hint, actual count may differ after deduplication).
         * @param reverse_edges If true, edges are treated as (destination, source).
         * @param bv_factory Factory for creating bit vectors.
         */
        U_GEF_AdjList(
            const std::vector<std::pair<size_t, size_t>>& edges,
            size_t n,
            size_t m,
            bool reverse_edges = false,
            std::shared_ptr<IBitVectorFactory> bv_factory = nullptr
        ) : n_(n), m_(0), base_(0), b_(0), h_(0), non_empty_count_(0) {
            
            if (n == 0 && m > 0) {
                throw std::invalid_argument("Cannot have edges with 0 nodes");
            }

            if (!bv_factory) {
                bv_factory = std::make_shared<SDSLBitVectorFactory>();
            }

            if (n == 0 || edges.empty()) {
                std::vector<size_t> offset_values(n + 1, 0);
                offsets_ = MyEF<size_t, BitVectorType>(bv_factory, offset_values, OPTIMAL_SPLIT_POINT);
                return;
            }

            // Phase 1: Create sorted edge list (source, dest) with optional reversal
            // This is the only copy of edge data we need
            std::vector<std::pair<T, T>> sorted_edges;
            sorted_edges.reserve(edges.size());
            
            for (const auto& edge : edges) {
                size_t u = edge.first;
                size_t v = edge.second;
                if (reverse_edges) std::swap(u, v);
                if (u < n && v < n) {
                    sorted_edges.emplace_back(static_cast<T>(u), static_cast<T>(v));
                }
            }
            
            // Sort by (source, dest) for grouping and deduplication
            std::sort(sorted_edges.begin(), sorted_edges.end());
            
            // Remove duplicates
            sorted_edges.erase(std::unique(sorted_edges.begin(), sorted_edges.end()), sorted_edges.end());
            sorted_edges.shrink_to_fit();
            
            m_ = sorted_edges.size();
            
            if (m_ == 0) {
                std::vector<size_t> offset_values(n + 1, 0);
                offsets_ = MyEF<size_t, BitVectorType>(bv_factory, offset_values, OPTIMAL_SPLIT_POINT);
                return;
            }

            // Phase 2: Compute offsets, min/max, non_empty_count in single pass
            std::vector<size_t> offset_values(n + 1, 0);
            T min_val = sorted_edges[0].second;
            T max_val = sorted_edges[0].second;
            
            for (const auto& [src, dst] : sorted_edges) {
                offset_values[src + 1]++;
                if (dst < min_val) min_val = dst;
                if (dst > max_val) max_val = dst;
            }
            
            // Convert counts to cumulative offsets
            for (size_t i = 1; i <= n; ++i) {
                if (offset_values[i] > 0) ++non_empty_count_;
                offset_values[i] += offset_values[i - 1];
            }

            base_ = min_val;
            const uint8_t total_bits = bits_for_range(base_, max_val);
            b_ = compute_split_point(m_, base_, max_val);
            if (b_ > total_bits) b_ = total_bits;
            h_ = (b_ >= total_bits) ? 0 : static_cast<uint8_t>(total_bits - b_);

            using U = std::make_unsigned_t<T>;
            const U low_mask = (b_ > 0) ? ((U(1) << b_) - 1) : 0;

            // Phase 3: Compute g_bits by iterating through sorted edges
            size_t g_bits = 0;
            if (h_ > 0) {
                T current_src = sorted_edges[0].first;
                U prev_high = (static_cast<U>(sorted_edges[0].second) - static_cast<U>(base_)) >> b_;
                
                for (size_t i = 1; i < m_; ++i) {
                    T src = sorted_edges[i].first;
                    U curr_high = (static_cast<U>(sorted_edges[i].second) - static_cast<U>(base_)) >> b_;
                    
                    if (src == current_src) {
                        // Same source node - compute gap
                        g_bits += (curr_high - prev_high) + 1;
                    }
                    // else: new source node, this will be an exception (no gap)
                    
                    current_src = src;
                    prev_high = curr_high;
                }
            }

            // Phase 4: Allocate final structures
            offsets_ = MyEF<size_t, BitVectorType>(bv_factory, offset_values, OPTIMAL_SPLIT_POINT);
            std::vector<size_t>().swap(offset_values); // Release memory

            if (b_ > 0) {
                L_ = sdsl::int_vector<>(m_, 0, b_);
            } else {
                L_ = sdsl::int_vector<>(0);
            }

            // Handle h_ == 0 case
            if (h_ == 0) {
                for (size_t i = 0; i < m_; ++i) {
                    if (b_ > 0) {
                        L_[i] = static_cast<typename sdsl::int_vector<>::value_type>(
                            static_cast<U>(sorted_edges[i].second) - static_cast<U>(base_)
                        );
                    }
                }
                G_ = nullptr;
                H_.resize(0);
                return;
            }

            H_ = sdsl::int_vector<>(non_empty_count_, 0, h_);
            G_ = std::make_unique<BitVectorType>(g_bits > 0 ? g_bits : 1);

            // Phase 5: Populate L, H, G in single pass through sorted edges
            uint64_t* g_data = G_->raw_data_ptr();
            FastBitWriter<BitVectorType::reverse_bit_order> g_writer(g_data);

            size_t h_idx = 0;
            T current_src = sorted_edges[0].first;
            
            // First edge is always an exception
            U first_val = static_cast<U>(sorted_edges[0].second) - static_cast<U>(base_);
            if (b_ > 0) {
                L_[0] = static_cast<typename sdsl::int_vector<>::value_type>(first_val & low_mask);
            }
            U prev_high = first_val >> b_;
            H_[h_idx++] = prev_high;

            for (size_t i = 1; i < m_; ++i) {
                T src = sorted_edges[i].first;
                U val = static_cast<U>(sorted_edges[i].second) - static_cast<U>(base_);
                
                if (b_ > 0) {
                    L_[i] = static_cast<typename sdsl::int_vector<>::value_type>(val & low_mask);
                }
                
                U curr_high = val >> b_;
                
                if (src != current_src) {
                    // New source node - this is an exception
                    H_[h_idx++] = curr_high;
                    current_src = src;
                } else {
                    // Same source node - encode gap
                    U gap = curr_high - prev_high;
                    g_writer.set_ones_range(static_cast<uint64_t>(gap));
                    g_writer.set_zero();
                }
                
                prev_high = curr_high;
            }

            // Enable select support for G
            if (g_bits > 0) {
                G_->enable_select0();
            }
        }

        /**
         * @brief Move-enabled constructor that takes ownership of edges.
         * 
         * This constructor avoids copying edges by taking them by value,
         * which enables move semantics when called with std::move(edges).
         * The edges are sorted in-place to minimize memory usage.
         * 
         * @param edges Vector of edges (will be moved/modified).
         * @param n Number of nodes.
         * @param reverse_edges If true, edges are treated as (destination, source).
         * @param bv_factory Factory for creating bit vectors.
         */
        U_GEF_AdjList(
            std::vector<std::pair<size_t, size_t>> edges,  // By value for move semantics
            size_t n,
            bool reverse_edges,
            std::shared_ptr<IBitVectorFactory> bv_factory
        ) : n_(n), m_(0), base_(0), b_(0), h_(0), non_empty_count_(0) {
            
            if (!bv_factory) {
                bv_factory = std::make_shared<SDSLBitVectorFactory>();
            }

            if (n == 0 || edges.empty()) {
                std::vector<size_t> offset_values(n + 1, 0);
                offsets_ = MyEF<size_t, BitVectorType>(bv_factory, offset_values, OPTIMAL_SPLIT_POINT);
                return;
            }

            // Apply reversal in-place if needed
            if (reverse_edges) {
                for (auto& e : edges) {
                    std::swap(e.first, e.second);
                }
            }

            // Filter out-of-bounds edges in-place
            edges.erase(
                std::remove_if(edges.begin(), edges.end(),
                    [n](const auto& e) { return e.first >= n || e.second >= n; }),
                edges.end()
            );
            
            // Sort by (source, dest) for grouping and deduplication - in place!
            std::sort(edges.begin(), edges.end());
            
            // Remove duplicates in-place
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
            edges.shrink_to_fit();
            
            m_ = edges.size();
            
            if (m_ == 0) {
                std::vector<size_t> offset_values(n + 1, 0);
                offsets_ = MyEF<size_t, BitVectorType>(bv_factory, offset_values, OPTIMAL_SPLIT_POINT);
                return;
            }

            // Compute offsets, min/max, non_empty_count
            std::vector<size_t> offset_values(n + 1, 0);
            T min_val = static_cast<T>(edges[0].second);
            T max_val = static_cast<T>(edges[0].second);
            
            for (const auto& [src, dst] : edges) {
                offset_values[src + 1]++;
                T dst_t = static_cast<T>(dst);
                if (dst_t < min_val) min_val = dst_t;
                if (dst_t > max_val) max_val = dst_t;
            }
            
            for (size_t i = 1; i <= n; ++i) {
                if (offset_values[i] > 0) ++non_empty_count_;
                offset_values[i] += offset_values[i - 1];
            }

            base_ = min_val;
            const uint8_t total_bits = bits_for_range(base_, max_val);
            b_ = compute_split_point(m_, base_, max_val);
            if (b_ > total_bits) b_ = total_bits;
            h_ = (b_ >= total_bits) ? 0 : static_cast<uint8_t>(total_bits - b_);

            using U = std::make_unsigned_t<T>;
            const U low_mask = (b_ > 0) ? ((U(1) << b_) - 1) : 0;

            // Compute g_bits
            size_t g_bits = 0;
            if (h_ > 0) {
                size_t current_src = edges[0].first;
                U prev_high = (static_cast<U>(edges[0].second) - static_cast<U>(base_)) >> b_;
                
                for (size_t i = 1; i < m_; ++i) {
                    size_t src = edges[i].first;
                    U curr_high = (static_cast<U>(edges[i].second) - static_cast<U>(base_)) >> b_;
                    
                    if (src == current_src) {
                        g_bits += (curr_high - prev_high) + 1;
                    }
                    current_src = src;
                    prev_high = curr_high;
                }
            }

            // Allocate final structures
            offsets_ = MyEF<size_t, BitVectorType>(bv_factory, offset_values, OPTIMAL_SPLIT_POINT);
            std::vector<size_t>().swap(offset_values);

            if (b_ > 0) {
                L_ = sdsl::int_vector<>(m_, 0, b_);
            } else {
                L_ = sdsl::int_vector<>(0);
            }

            if (h_ == 0) {
                for (size_t i = 0; i < m_; ++i) {
                    if (b_ > 0) {
                        L_[i] = static_cast<typename sdsl::int_vector<>::value_type>(
                            static_cast<U>(edges[i].second) - static_cast<U>(base_)
                        );
                    }
                }
                G_ = nullptr;
                H_.resize(0);
                return;
            }

            H_ = sdsl::int_vector<>(non_empty_count_, 0, h_);
            G_ = std::make_unique<BitVectorType>(g_bits > 0 ? g_bits : 1);

            // Populate L, H, G
            uint64_t* g_data = G_->raw_data_ptr();
            FastBitWriter<BitVectorType::reverse_bit_order> g_writer(g_data);

            size_t h_idx = 0;
            size_t current_src = edges[0].first;
            
            U first_val = static_cast<U>(edges[0].second) - static_cast<U>(base_);
            if (b_ > 0) {
                L_[0] = static_cast<typename sdsl::int_vector<>::value_type>(first_val & low_mask);
            }
            U prev_high = first_val >> b_;
            H_[h_idx++] = prev_high;

            for (size_t i = 1; i < m_; ++i) {
                size_t src = edges[i].first;
                U val = static_cast<U>(edges[i].second) - static_cast<U>(base_);
                
                if (b_ > 0) {
                    L_[i] = static_cast<typename sdsl::int_vector<>::value_type>(val & low_mask);
                }
                
                U curr_high = val >> b_;
                
                if (src != current_src) {
                    H_[h_idx++] = curr_high;
                    current_src = src;
                } else {
                    U gap = curr_high - prev_high;
                    g_writer.set_ones_range(static_cast<uint64_t>(gap));
                    g_writer.set_zero();
                }
                
                prev_high = curr_high;
            }

            if (g_bits > 0) {
                G_->enable_select0();
            }
        }

        // Copy constructor
        U_GEF_AdjList(const U_GEF_AdjList& other)
            : n_(other.n_),
              m_(other.m_),
              base_(other.base_),
              b_(other.b_),
              h_(other.h_),
              offsets_(other.offsets_),
              H_(other.H_),
              L_(other.L_),
              non_empty_count_(other.non_empty_count_) {
            if (other.G_) {
                G_ = std::make_unique<BitVectorType>(*other.G_);
                G_->enable_select0();
            }
        }

        // Copy assignment
        U_GEF_AdjList& operator=(const U_GEF_AdjList& other) {
            if (this != &other) {
                n_ = other.n_;
                m_ = other.m_;
                base_ = other.base_;
                b_ = other.b_;
                h_ = other.h_;
                offsets_ = other.offsets_;
                H_ = other.H_;
                L_ = other.L_;
                non_empty_count_ = other.non_empty_count_;
                if (other.G_) {
                    G_ = std::make_unique<BitVectorType>(*other.G_);
                    G_->enable_select0();
                } else {
                    G_ = nullptr;
                }
            }
            return *this;
        }

        // Move constructor
        U_GEF_AdjList(U_GEF_AdjList&& other) noexcept
            : n_(other.n_),
              m_(other.m_),
              base_(other.base_),
              b_(other.b_),
              h_(other.h_),
              offsets_(std::move(other.offsets_)),
              G_(std::move(other.G_)),
              H_(std::move(other.H_)),
              L_(std::move(other.L_)),
              non_empty_count_(other.non_empty_count_) {
            other.n_ = 0;
            other.m_ = 0;
            other.non_empty_count_ = 0;
        }

        // Move assignment
        U_GEF_AdjList& operator=(U_GEF_AdjList&& other) noexcept {
            if (this != &other) {
                n_ = other.n_;
                m_ = other.m_;
                base_ = other.base_;
                b_ = other.b_;
                h_ = other.h_;
                offsets_ = std::move(other.offsets_);
                G_ = std::move(other.G_);
                H_ = std::move(other.H_);
                L_ = std::move(other.L_);
                non_empty_count_ = other.non_empty_count_;
                other.n_ = 0;
                other.m_ = 0;
                other.non_empty_count_ = 0;
            }
            return *this;
        }

        /**
         * @brief Returns the i-th neighbor of the given node.
         * 
         * @param node_index The node ID.
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

            return decodeElement(node_index, i, start, end);
        }

        /**
         * @brief Returns all neighbors of the given node.
         * 
         * @param node_index The node ID.
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

            std::vector<T> result(count);
            decodeRange(node_index, start, end, result);
            return result;
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
            ofs.write(reinterpret_cast<const char*>(&base_), sizeof(base_));
            ofs.write(reinterpret_cast<const char*>(&b_), sizeof(b_));
            ofs.write(reinterpret_cast<const char*>(&h_), sizeof(h_));
            ofs.write(reinterpret_cast<const char*>(&non_empty_count_), sizeof(non_empty_count_));
            
            offsets_.serialize(ofs);
            
            if (b_ > 0) {
                L_.serialize(ofs);
            }
            
            H_.serialize(ofs);
            
            bool has_g = (G_ != nullptr);
            ofs.write(reinterpret_cast<const char*>(&has_g), sizeof(has_g));
            if (has_g) {
                G_->serialize(ofs);
            }
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
            ifs.read(reinterpret_cast<char*>(&base_), sizeof(base_));
            ifs.read(reinterpret_cast<char*>(&b_), sizeof(b_));
            ifs.read(reinterpret_cast<char*>(&h_), sizeof(h_));
            ifs.read(reinterpret_cast<char*>(&non_empty_count_), sizeof(non_empty_count_));
            
            offsets_.load(ifs, bv_factory);
            
            if (b_ > 0) {
                L_.load(ifs);
            } else {
                L_ = sdsl::int_vector<>(0);
            }
            
            H_.load(ifs);
            
            bool has_g;
            ifs.read(reinterpret_cast<char*>(&has_g), sizeof(has_g));
            if (has_g) {
                G_ = std::make_unique<BitVectorType>(BitVectorType::load(ifs));
                G_->enable_select0();
            } else {
                G_ = nullptr;
            }
        }

        /**
         * @brief Returns total size in bytes.
         */
        size_t size_in_bytes() const {
            size_t total = sizeof(n_) + sizeof(m_) + sizeof(base_) + sizeof(b_) + sizeof(h_) + sizeof(non_empty_count_);
            total += offsets_.size_in_bytes();
            total += sdsl::size_in_bytes(L_);
            total += sdsl::size_in_bytes(H_);
            if (G_) {
                total += G_->size_in_bytes();
            }
            return total;
        }

        size_t num_nodes() const { return n_; }
        size_t num_edges() const { return m_; }

    private:
        /**
         * @brief Decodes a single element at position (node_index, i) within its neighborhood.
         */
        T decodeElement(size_t node_index, size_t local_idx, size_t start, size_t end) const {
            using U = std::make_unsigned_t<T>;
            
            size_t global_idx = start + local_idx;
            
            // Get low part
            U low = (b_ > 0) ? static_cast<U>(L_[global_idx]) : 0;
            
            // If h_ == 0, all info is in low part
            if (h_ == 0) {
                return base_ + static_cast<T>(low);
            }
            
            // Find which exception (H index) this node corresponds to
            size_t h_idx = countNonEmptyBefore(node_index);
            U current_high = static_cast<U>(H_[h_idx]);
            
            if (local_idx == 0) {
                // First element - use H directly
                return base_ + static_cast<T>(low | (current_high << b_));
            }
            
            // Need to decode gaps from position 1 to local_idx
            // Find the starting bit position in G for this neighborhood
            size_t g_start_bit = computeGStartBit(node_index);
            
            // Decode gaps sequentially
            FastUnaryDecoder<BitVectorType::reverse_bit_order> decoder(G_->raw_data_ptr(), G_->size(), g_start_bit);
            
            for (size_t k = 1; k <= local_idx; ++k) {
                uint64_t gap = decoder.next();
                current_high += static_cast<U>(gap);
            }
            
            return base_ + static_cast<T>(low | (current_high << b_));
        }

        /**
         * @brief Decodes all elements in a range [start, end) for a given node.
         */
        void decodeRange(size_t node_index, size_t start, size_t end, std::vector<T>& output) const {
            using U = std::make_unsigned_t<T>;
            size_t count = end - start;
            
            if (h_ == 0) {
                // All data in L
                for (size_t i = 0; i < count; ++i) {
                    U low = (b_ > 0) ? static_cast<U>(L_[start + i]) : 0;
                    output[i] = base_ + static_cast<T>(low);
                }
                return;
            }
            
            // Find H index for this node
            size_t h_idx = countNonEmptyBefore(node_index);
            U current_high = static_cast<U>(H_[h_idx]);
            
            // First element
            U low = (b_ > 0) ? static_cast<U>(L_[start]) : 0;
            output[0] = base_ + static_cast<T>(low | (current_high << b_));
            
            if (count == 1) return;
            
            // Find G starting bit for this neighborhood
            size_t g_start_bit = computeGStartBit(node_index);
            
            // Decode remaining elements using gap decoder
            FastUnaryDecoder<BitVectorType::reverse_bit_order> decoder(G_->raw_data_ptr(), G_->size(), g_start_bit);
            
            constexpr size_t GAP_BATCH = 64;
            uint32_t gap_buffer[GAP_BATCH];
            size_t buffer_size = 0;
            size_t buffer_index = 0;
            
            for (size_t i = 1; i < count; ++i) {
                if (buffer_index >= buffer_size) {
                    buffer_size = decoder.next_batch(gap_buffer, GAP_BATCH);
                    buffer_index = 0;
                    if (buffer_size == 0) {
                        gap_buffer[0] = static_cast<uint32_t>(decoder.next());
                        buffer_size = 1;
                    }
                }
                
                current_high += static_cast<U>(gap_buffer[buffer_index++]);
                low = (b_ > 0) ? static_cast<U>(L_[start + i]) : 0;
                output[i] = base_ + static_cast<T>(low | (current_high << b_));
            }
        }

        /**
         * @brief Count how many nodes before node_index have non-empty neighborhoods.
         * This gives us the index into H for node_index's first neighbor.
         */
        size_t countNonEmptyBefore(size_t node_index) const {
            // We need to count how many nodes i < node_index have degree > 0
            // This is equivalent to counting how many offsets_[i] < offsets_[i+1]
            size_t count = 0;
            for (size_t i = 0; i < node_index; ++i) {
                if (offsets_[i] < offsets_[i + 1]) {
                    ++count;
                }
            }
            return count;
        }

        /**
         * @brief Compute the starting bit position in G for the gaps of node_index's neighborhood.
         * 
         * G stores gaps for all non-first elements. For each node i < node_index with degree d_i > 1,
         * we have (d_i - 1) gaps encoded. We need to sum the bit lengths of all previous gaps.
         */
        size_t computeGStartBit(size_t node_index) const {
            // Count total non-first elements before this node's neighborhood
            size_t non_first_before = 0;
            for (size_t i = 0; i < node_index; ++i) {
                size_t deg = offsets_[i + 1] - offsets_[i];
                if (deg > 1) {
                    non_first_before += (deg - 1);
                }
            }
            
            if (non_first_before == 0) {
                return 0;
            }
            
            // Position after non_first_before zeros (terminators) in G
            // select0(k) returns position of k-th zero (1-indexed typically)
            return G_->select0(non_first_before) + 1;
        }
    };

} // namespace gef

#endif // U_GEF_ADJ_LIST_HPP

