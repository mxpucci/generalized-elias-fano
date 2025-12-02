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
#include <cmath>
#include <random>
#include <numeric>

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
         * @brief Computes Reverse Cuthill-McKee (RCM) ordering of graph nodes.
         * 
         * RCM ordering is a bandwidth-reducing ordering that:
         * 1. Starts from a peripheral node (minimum degree node)
         * 2. Visits neighbors in order of increasing degree
         * 3. Reverses the final ordering
         * 
         * This tends to produce better locality than simple BFS ordering.
         * 
         * @param edges Original edges
         * @param num_nodes Number of nodes in the graph
         * @return Pair of mappings: (original_to_new, new_to_original)
         */
        static std::pair<std::vector<size_t>, std::vector<size_t>>
        compute_rcm_ordering(const std::vector<std::pair<size_t, size_t>>& edges, size_t num_nodes) {
            // Build undirected adjacency list
            std::vector<std::vector<size_t>> adj(num_nodes);
            std::vector<size_t> degree(num_nodes, 0);
            std::unordered_set<size_t> nodes_in_edges;

            for (const auto& [u, v] : edges) {
                if (u < num_nodes && v < num_nodes) {
                    adj[u].push_back(v);
                    adj[v].push_back(u);
                    nodes_in_edges.insert(u);
                    nodes_in_edges.insert(v);
                }
            }

            // Remove duplicates and compute degrees
            for (size_t i = 0; i < num_nodes; ++i) {
                std::sort(adj[i].begin(), adj[i].end());
                adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
                degree[i] = adj[i].size();
            }

            // Sort adjacency lists by degree (ascending) for Cuthill-McKee
            for (size_t i = 0; i < num_nodes; ++i) {
                std::sort(adj[i].begin(), adj[i].end(), 
                    [&degree](size_t a, size_t b) { return degree[a] < degree[b]; });
            }

            // Cuthill-McKee ordering (will be reversed at the end)
            std::vector<size_t> cm_order;
            cm_order.reserve(nodes_in_edges.size());
            std::vector<bool> visited(num_nodes, false);

            // Process all connected components
            while (cm_order.size() < nodes_in_edges.size()) {
                // Find the unvisited node with minimum degree (peripheral node)
                size_t start = SIZE_MAX;
                size_t min_degree = SIZE_MAX;
                for (size_t i = 0; i < num_nodes; ++i) {
                    if (!visited[i] && nodes_in_edges.count(i) > 0) {
                        if (degree[i] < min_degree) {
                            min_degree = degree[i];
                            start = i;
                        }
                    }
                }

                if (start == SIZE_MAX) break;

                // BFS-like traversal with degree-sorted neighbors
                std::queue<size_t> q;
                q.push(start);
                visited[start] = true;

                while (!q.empty()) {
                    size_t current = q.front();
                    q.pop();
                    cm_order.push_back(current);

                    // Add unvisited neighbors (already sorted by degree)
                    for (size_t neighbor : adj[current]) {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }

            // Reverse the ordering (Cuthill-McKee -> Reverse Cuthill-McKee)
            std::reverse(cm_order.begin(), cm_order.end());

            // Create mappings
            std::vector<size_t> original_to_new(num_nodes, SIZE_MAX);
            std::vector<size_t> new_to_original;
            new_to_original.reserve(cm_order.size());

            for (size_t new_id = 0; new_id < cm_order.size(); ++new_id) {
                size_t orig_id = cm_order[new_id];
                original_to_new[orig_id] = new_id;
                new_to_original.push_back(orig_id);
            }

            return {std::move(original_to_new), std::move(new_to_original)};
        }

        /**
         * @brief Computes spectral (Fiedler vector) ordering of graph nodes.
         * 
         * Orders nodes according to the Fiedler vector - the eigenvector corresponding
         * to the second smallest eigenvalue of the graph Laplacian. This ordering tends
         * to place connected nodes close together, which can improve compression.
         * 
         * Uses power iteration with deflation to compute the Fiedler vector.
         * 
         * @param edges Original edges
         * @param num_nodes Number of nodes in the graph
         * @param max_iterations Maximum number of power iterations (default: 100)
         * @param tolerance Convergence tolerance (default: 1e-6)
         * @return Pair of mappings: (original_to_new, new_to_original)
         */
        static std::pair<std::vector<size_t>, std::vector<size_t>>
        compute_spectral_ordering(const std::vector<std::pair<size_t, size_t>>& edges, 
                                  size_t num_nodes,
                                  size_t max_iterations = 100,
                                  double tolerance = 1e-6) {
            // Build undirected adjacency list and compute degrees
            std::vector<std::vector<size_t>> adj(num_nodes);
            std::vector<size_t> degree(num_nodes, 0);
            std::unordered_set<size_t> nodes_in_edges;

            for (const auto& [u, v] : edges) {
                if (u < num_nodes && v < num_nodes) {
                    adj[u].push_back(v);
                    adj[v].push_back(u);
                    nodes_in_edges.insert(u);
                    nodes_in_edges.insert(v);
                }
            }

            // Remove duplicate edges and compute degrees
            for (size_t i = 0; i < num_nodes; ++i) {
                std::sort(adj[i].begin(), adj[i].end());
                adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
                degree[i] = adj[i].size();
            }

            // Collect active nodes (nodes that appear in edges)
            std::vector<size_t> active_nodes;
            active_nodes.reserve(nodes_in_edges.size());
            for (size_t i = 0; i < num_nodes; ++i) {
                if (nodes_in_edges.count(i) > 0) {
                    active_nodes.push_back(i);
                }
            }

            if (active_nodes.empty()) {
                return {{}, {}};
            }

            const size_t n_active = active_nodes.size();
            
            // Create mapping from original to active index
            std::vector<size_t> orig_to_active(num_nodes, SIZE_MAX);
            for (size_t i = 0; i < n_active; ++i) {
                orig_to_active[active_nodes[i]] = i;
            }

            // Initialize Fiedler vector approximation with random values
            std::vector<double> x(n_active);
            std::mt19937 rng(42);  // Fixed seed for reproducibility
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (size_t i = 0; i < n_active; ++i) {
                x[i] = dist(rng);
            }

            // Normalize and orthogonalize against constant vector
            auto orthogonalize_and_normalize = [&](std::vector<double>& v) {
                // Remove mean (orthogonalize against constant vector)
                double mean = 0.0;
                for (double val : v) mean += val;
                mean /= static_cast<double>(v.size());
                for (double& val : v) val -= mean;

                // Normalize
                double norm = 0.0;
                for (double val : v) norm += val * val;
                norm = std::sqrt(norm);
                if (norm > 1e-10) {
                    for (double& val : v) val /= norm;
                }
            };

            orthogonalize_and_normalize(x);

            // Power iteration on (I - L/d_max) to find Fiedler vector
            // where L = D - A is the Laplacian
            // This converges to the eigenvector with LARGEST eigenvalue of (I - L/d_max)
            // which corresponds to SMALLEST non-zero eigenvalue of L
            
            // Find max degree for normalization
            size_t max_degree = 1;
            for (size_t node : active_nodes) {
                max_degree = std::max(max_degree, degree[node]);
            }
            const double d_max = static_cast<double>(max_degree);

            std::vector<double> y(n_active);

            for (size_t iter = 0; iter < max_iterations; ++iter) {
                // Compute y = (I - L/d_max) * x = x - (D - A)*x / d_max
                // = x - D*x/d_max + A*x/d_max
                for (size_t i = 0; i < n_active; ++i) {
                    size_t orig_node = active_nodes[i];
                    double deg_i = static_cast<double>(degree[orig_node]);
                    
                    // A*x contribution for node i
                    double ax_i = 0.0;
                    for (size_t neighbor : adj[orig_node]) {
                        size_t neighbor_active = orig_to_active[neighbor];
                        if (neighbor_active != SIZE_MAX) {
                            ax_i += x[neighbor_active];
                        }
                    }
                    
                    // y[i] = x[i] - deg_i * x[i] / d_max + ax_i / d_max
                    y[i] = x[i] * (1.0 - deg_i / d_max) + ax_i / d_max;
                }

                // Orthogonalize against constant vector and normalize
                orthogonalize_and_normalize(y);

                // Check convergence
                double diff = 0.0;
                for (size_t i = 0; i < n_active; ++i) {
                    double d = y[i] - x[i];
                    diff += d * d;
                }
                
                x.swap(y);
                
                if (std::sqrt(diff) < tolerance) {
                    break;
                }
            }

            // Sort active nodes by their Fiedler vector values
            std::vector<std::pair<double, size_t>> fiedler_order(n_active);
            for (size_t i = 0; i < n_active; ++i) {
                fiedler_order[i] = {x[i], active_nodes[i]};
            }
            std::sort(fiedler_order.begin(), fiedler_order.end());

            // Create mappings
            std::vector<size_t> original_to_new(num_nodes, SIZE_MAX);
            std::vector<size_t> new_to_original;
            new_to_original.reserve(n_active);

            for (size_t new_id = 0; new_id < n_active; ++new_id) {
                size_t orig_id = fiedler_order[new_id].second;
                original_to_new[orig_id] = new_id;
                new_to_original.push_back(orig_id);
            }

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
         * @brief Full preprocessing pipeline with Reverse Cuthill-McKee ordering.
         * 
         * @param filepath Path to the SNAP-format edge list file
         * @return PreprocessedGraph with RCM-relabeled edges and mappings
         */
        static PreprocessedGraph preprocess_rcm(const std::filesystem::path& filepath) {
            // 1. Parse the file
            auto [edges, num_nodes_original] = parse_snap_file(filepath);

            // 2. Compute RCM ordering on undirected graph
            auto [original_to_new, new_to_original] = compute_rcm_ordering(edges, num_nodes_original);

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
         * @brief Preprocess with RCM ordering and custom number of nodes.
         * 
         * @param filepath Path to the SNAP-format edge list file
         * @param expected_nodes Expected number of nodes (from file metadata)
         * @return PreprocessedGraph with RCM-relabeled edges and mappings
         */
        static PreprocessedGraph preprocess_rcm(const std::filesystem::path& filepath, 
                                                size_t expected_nodes) {
            // 1. Parse the file
            auto [edges, num_nodes_from_edges] = parse_snap_file(filepath);
            
            // Use the larger of expected or computed
            size_t num_nodes = std::max(expected_nodes, num_nodes_from_edges);

            // 2. Compute RCM ordering on undirected graph
            auto [original_to_new, new_to_original] = compute_rcm_ordering(edges, num_nodes);

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
         * @brief Full preprocessing pipeline with spectral (Fiedler) ordering.
         * 
         * @param filepath Path to the SNAP-format edge list file
         * @return PreprocessedGraph with spectrally-relabeled edges and mappings
         */
        static PreprocessedGraph preprocess_spectral(const std::filesystem::path& filepath) {
            // 1. Parse the file
            auto [edges, num_nodes_original] = parse_snap_file(filepath);

            // 2. Compute spectral ordering on undirected graph
            auto [original_to_new, new_to_original] = compute_spectral_ordering(edges, num_nodes_original);

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
         * @brief Preprocess with spectral ordering and custom number of nodes.
         * 
         * @param filepath Path to the SNAP-format edge list file
         * @param expected_nodes Expected number of nodes (from file metadata)
         * @return PreprocessedGraph with spectrally-relabeled edges and mappings
         */
        static PreprocessedGraph preprocess_spectral(const std::filesystem::path& filepath, 
                                                     size_t expected_nodes) {
            // 1. Parse the file
            auto [edges, num_nodes_from_edges] = parse_snap_file(filepath);
            
            // Use the larger of expected or computed
            size_t num_nodes = std::max(expected_nodes, num_nodes_from_edges);

            // 2. Compute spectral ordering on undirected graph
            auto [original_to_new, new_to_original] = compute_spectral_ordering(edges, num_nodes);

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



