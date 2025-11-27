//
// Created by Michelangelo Pucci on 26/11/25.
//

#ifndef GRAPH_PREPROCESSOR_HPP
#define GRAPH_PREPROCESSOR_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>

namespace gef {

    /**
     * @brief Result of graph preprocessing containing edges and metadata.
     */
    struct PreprocessedGraph {
        std::vector<std::pair<size_t, size_t>> edges;  // Relabeled edges
        size_t num_nodes;
        size_t num_edges;
        std::vector<size_t> original_to_new;  // Mapping: original node ID -> new BFS-ordered ID
        std::vector<size_t> new_to_original;  // Mapping: new BFS-ordered ID -> original node ID
    };

    /**
     * @brief Preprocessor for graph datasets in SNAP format.
     * 
     * Supports parsing edge list files and relabeling nodes according to BFS
     * visit order on the undirected version of the graph.
     */
    class GraphPreprocessor {
    public:
        /**
         * @brief Parses a SNAP-format edge list file.
         * 
         * SNAP format:
         * - Lines starting with '#' are comments
         * - Data lines have format: FromNodeId<tab>ToNodeId
         * 
         * @param filepath Path to the edge list file
         * @return Pair of (edges, max_node_id + 1)
         */
        static std::pair<std::vector<std::pair<size_t, size_t>>, size_t> 
        parse_snap_file(const std::filesystem::path& filepath) {
            std::ifstream ifs(filepath);
            if (!ifs.is_open()) {
                throw std::runtime_error("Failed to open file: " + filepath.string());
            }

            std::vector<std::pair<size_t, size_t>> edges;
            size_t max_node_id = 0;
            std::string line;

            while (std::getline(ifs, line)) {
                // Skip empty lines and comments
                if (line.empty() || line[0] == '#') {
                    continue;
                }

                std::istringstream iss(line);
                size_t u, v;
                if (iss >> u >> v) {
                    edges.emplace_back(u, v);
                    max_node_id = std::max(max_node_id, std::max(u, v));
                }
            }

            return {std::move(edges), max_node_id + 1};
        }

        /**
         * @brief Computes BFS ordering starting from the node with minimum ID.
         * 
         * Creates a mapping where nodes are relabeled according to BFS visit order
         * on the undirected version of the graph. Handles disconnected components
         * by starting new BFS from the smallest unvisited node.
         * 
         * @param edges Original edges
         * @param num_nodes Number of nodes in the graph
         * @return Pair of mappings: (original_to_new, new_to_original)
         */
        static std::pair<std::vector<size_t>, std::vector<size_t>>
        compute_bfs_ordering(const std::vector<std::pair<size_t, size_t>>& edges, size_t num_nodes) {
            // Build undirected adjacency list
            std::vector<std::vector<size_t>> adj(num_nodes);
            std::unordered_set<size_t> nodes_in_edges;

            for (const auto& [u, v] : edges) {
                if (u < num_nodes && v < num_nodes) {
                    adj[u].push_back(v);
                    adj[v].push_back(u);
                    nodes_in_edges.insert(u);
                    nodes_in_edges.insert(v);
                }
            }

            // Sort adjacency lists for deterministic BFS
            for (auto& neighbors : adj) {
                std::sort(neighbors.begin(), neighbors.end());
                // Remove duplicates
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            }

            // BFS to compute ordering
            std::vector<size_t> original_to_new(num_nodes, SIZE_MAX);
            std::vector<size_t> new_to_original;
            new_to_original.reserve(num_nodes);

            std::vector<bool> visited(num_nodes, false);
            size_t new_id = 0;

            // Process all connected components
            for (size_t start = 0; start < num_nodes; ++start) {
                // Skip already visited nodes and isolated nodes not in any edge
                if (visited[start]) continue;
                if (nodes_in_edges.find(start) == nodes_in_edges.end()) continue;

                std::queue<size_t> bfs_queue;
                bfs_queue.push(start);
                visited[start] = true;

                while (!bfs_queue.empty()) {
                    size_t current = bfs_queue.front();
                    bfs_queue.pop();

                    // Assign new ID
                    original_to_new[current] = new_id;
                    new_to_original.push_back(current);
                    ++new_id;

                    // Visit neighbors in sorted order
                    for (size_t neighbor : adj[current]) {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            bfs_queue.push(neighbor);
                        }
                    }
                }
            }

            new_to_original.shrink_to_fit();
            return {std::move(original_to_new), std::move(new_to_original)};
        }

        /**
         * @brief Relabels edges according to a given mapping.
         * 
         * @param edges Original edges
         * @param original_to_new Mapping from original to new node IDs
         * @return Relabeled edges (edges with unmapped nodes are skipped)
         */
        static std::vector<std::pair<size_t, size_t>>
        relabel_edges(const std::vector<std::pair<size_t, size_t>>& edges,
                      const std::vector<size_t>& original_to_new) {
            std::vector<std::pair<size_t, size_t>> relabeled;
            relabeled.reserve(edges.size());

            for (const auto& [u, v] : edges) {
                if (u < original_to_new.size() && v < original_to_new.size() &&
                    original_to_new[u] != SIZE_MAX && original_to_new[v] != SIZE_MAX) {
                    relabeled.emplace_back(original_to_new[u], original_to_new[v]);
                }
            }

            return relabeled;
        }

        /**
         * @brief Full preprocessing pipeline: parse, compute BFS ordering, relabel.
         * 
         * @param filepath Path to the SNAP-format edge list file
         * @return PreprocessedGraph with relabeled edges and mappings
         */
        static PreprocessedGraph preprocess(const std::filesystem::path& filepath) {
            // 1. Parse the file
            auto [edges, num_nodes_original] = parse_snap_file(filepath);

            // 2. Compute BFS ordering on undirected graph
            auto [original_to_new, new_to_original] = compute_bfs_ordering(edges, num_nodes_original);

            // 3. Relabel edges
            auto relabeled_edges = relabel_edges(edges, original_to_new);

            // 4. Build result
            PreprocessedGraph result;
            result.edges = std::move(relabeled_edges);
            result.num_nodes = new_to_original.size();  // Only nodes that appear in edges
            result.num_edges = result.edges.size();
            result.original_to_new = std::move(original_to_new);
            result.new_to_original = std::move(new_to_original);

            return result;
        }

        /**
         * @brief Preprocess with custom number of nodes (useful when node count is known).
         * 
         * @param filepath Path to the SNAP-format edge list file
         * @param expected_nodes Expected number of nodes (from file metadata)
         * @return PreprocessedGraph with relabeled edges and mappings
         */
        static PreprocessedGraph preprocess(const std::filesystem::path& filepath, 
                                            size_t expected_nodes) {
            // 1. Parse the file
            auto [edges, num_nodes_from_edges] = parse_snap_file(filepath);
            
            // Use the larger of expected or computed
            size_t num_nodes = std::max(expected_nodes, num_nodes_from_edges);

            // 2. Compute BFS ordering on undirected graph
            auto [original_to_new, new_to_original] = compute_bfs_ordering(edges, num_nodes);

            // 3. Relabel edges
            auto relabeled_edges = relabel_edges(edges, original_to_new);

            // 4. Build result
            PreprocessedGraph result;
            result.edges = std::move(relabeled_edges);
            result.num_nodes = new_to_original.size();
            result.num_edges = result.edges.size();
            result.original_to_new = std::move(original_to_new);
            result.new_to_original = std::move(new_to_original);

            return result;
        }

        /**
         * @brief Parse metadata from SNAP file comments.
         * 
         * Extracts node and edge counts from comments like:
         * "# Nodes: 1965206 Edges: 5533214"
         * 
         * @param filepath Path to the SNAP-format file
         * @return Pair of (num_nodes, num_edges) or (0, 0) if not found
         */
        static std::pair<size_t, size_t> parse_metadata(const std::filesystem::path& filepath) {
            std::ifstream ifs(filepath);
            if (!ifs.is_open()) {
                return {0, 0};
            }

            size_t nodes = 0, edges = 0;
            std::string line;

            while (std::getline(ifs, line)) {
                if (line.empty() || line[0] != '#') {
                    break;  // Stop at first data line
                }

                // Look for "Nodes:" and "Edges:" patterns
                size_t nodes_pos = line.find("Nodes:");
                size_t edges_pos = line.find("Edges:");

                if (nodes_pos != std::string::npos) {
                    std::istringstream iss(line.substr(nodes_pos + 6));
                    iss >> nodes;
                }
                if (edges_pos != std::string::npos) {
                    std::istringstream iss(line.substr(edges_pos + 6));
                    iss >> edges;
                }
            }

            return {nodes, edges};
        }
    };

} // namespace gef

#endif // GRAPH_PREPROCESSOR_HPP


