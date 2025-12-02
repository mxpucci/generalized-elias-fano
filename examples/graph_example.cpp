//
// Example: Loading and querying compressed graphs
//
// This example demonstrates how to:
// 1. Load a SNAP-format graph dataset
// 2. Preprocess with BFS-based node relabeling
// 3. Build a compressed GEF_Graph with forward and reverse adjacency
// 4. Query neighbors efficiently
//

#include "../include/graphs/GEF_Graph.hpp"
#include "../include/graphs/GraphPreprocessor.hpp"
#include "../include/gef/B_GEF.hpp"
#include "../include/datastructures/PastaBitVectorFactory.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace gef;

void print_graph_stats(const auto& graph, const std::string& name) {
    std::cout << "\n=== " << name << " ===" << std::endl;
    std::cout << "Nodes: " << graph.num_nodes() << std::endl;
    std::cout << "Edges: " << graph.num_edges() << std::endl;
    std::cout << "Size: " << (graph.size_in_bytes() / (1024.0 * 1024.0)) 
              << " MB" << std::endl;
    std::cout << "Bits per edge: " << std::fixed << std::setprecision(2) 
              << graph.bits_per_edge() << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file.txt>" << std::endl;
        std::cerr << "\nExample SNAP files:" << std::endl;
        std::cerr << "  roadNet-CA.txt" << std::endl;
        std::cerr << "  soc-LiveJournal1.txt" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    std::cout << "Loading graph from: " << filepath << std::endl;

    // Create bit vector factory
    auto bv_factory = std::make_shared<PastaBitVectorFactory>();

    // Time the loading process
    auto start = std::chrono::steady_clock::now();

    // Option 1: Use the convenience constructor (handles preprocessing internally)
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(filepath, bv_factory);

    auto end = std::chrono::steady_clock::now();
    double load_time = std::chrono::duration<double>(end - start).count();

    std::cout << "Load time: " << load_time << " seconds" << std::endl;
    print_graph_stats(graph, "Compressed Graph");

    // Demonstrate queries
    std::cout << "\n=== Sample Queries ===" << std::endl;
    
    // Show first 5 nodes with their degrees
    size_t num_samples = std::min(static_cast<size_t>(5), graph.num_nodes());
    for (size_t i = 0; i < num_samples; ++i) {
        std::cout << "Node " << i << ": "
                  << "out_degree=" << graph.out_degree(i)
                  << ", in_degree=" << graph.in_degree(i);
        
        // Show original ID if available
        if (graph.has_original_mapping()) {
            std::cout << " (original ID: " << graph.to_original_id(i) << ")";
        }
        std::cout << std::endl;

        // Show first few neighbors
        auto out_nbrs = graph.out_neighbors(i);
        if (!out_nbrs.empty()) {
            std::cout << "  Out-neighbors: ";
            for (size_t j = 0; j < std::min(out_nbrs.size(), static_cast<size_t>(5)); ++j) {
                std::cout << out_nbrs[j] << " ";
            }
            if (out_nbrs.size() > 5) std::cout << "...";
            std::cout << std::endl;
        }
    }

    // Benchmark random access
    std::cout << "\n=== Random Access Benchmark ===" << std::endl;
    const size_t num_queries = 100000;
    
    start = std::chrono::steady_clock::now();
    size_t total_degree = 0;
    for (size_t i = 0; i < num_queries; ++i) {
        size_t node = (i * 31337) % graph.num_nodes();  // Pseudo-random
        total_degree += graph.out_degree(node);
    }
    end = std::chrono::steady_clock::now();
    
    double query_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Performed " << num_queries << " degree queries in " 
              << query_time << " seconds" << std::endl;
    std::cout << "Avg time per query: " << (query_time / num_queries * 1e6) 
              << " microseconds" << std::endl;
    std::cout << "(Total degree sum: " << total_degree << " to prevent optimization)" << std::endl;

    // Optional: Serialize the compressed graph
    std::string output_path = filepath + ".gef";
    std::cout << "\nSaving compressed graph to: " << output_path << std::endl;
    graph.serialize(output_path);
    
    std::cout << "Done!" << std::endl;
    return 0;
}






