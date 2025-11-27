//
// Tests for GEF_Graph and GraphPreprocessor
//

#include <gtest/gtest.h>
#include "../include/graphs/GEF_Graph.hpp"
#include "../include/graphs/GraphPreprocessor.hpp"
#include "../include/gef/B_GEF.hpp"
#include "../include/gef/MyEF.hpp"
#include "../include/datastructures/PastaBitVectorFactory.hpp"
#include "../include/datastructures/SDSLBitVectorFactory.hpp"

using namespace gef;

class GraphPreprocessorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

class GEFGraphTest : public ::testing::Test {
protected:
    std::shared_ptr<IBitVectorFactory> bv_factory;
    
    void SetUp() override {
        bv_factory = std::make_shared<PastaBitVectorFactory>();
    }
    void TearDown() override {}
};

// ===== GraphPreprocessor Tests =====

TEST_F(GraphPreprocessorTest, BFSOrderingSimpleGraph) {
    // Simple graph: 0->1->2->3->0 (cycle)
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}
    };
    
    auto [orig_to_new, new_to_orig] = GraphPreprocessor::compute_bfs_ordering(edges, 4);
    
    // All 4 nodes should be assigned IDs
    EXPECT_EQ(new_to_orig.size(), 4);
    
    // Each original node should have a valid new ID
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NE(orig_to_new[i], SIZE_MAX);
        EXPECT_LT(orig_to_new[i], 4);
    }
    
    // BFS from 0 should visit 0 first
    EXPECT_EQ(orig_to_new[0], 0);
}

TEST_F(GraphPreprocessorTest, BFSOrderingDisconnectedComponents) {
    // Two disconnected components: {0,1} and {2,3}
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {2, 3}
    };
    
    auto [orig_to_new, new_to_orig] = GraphPreprocessor::compute_bfs_ordering(edges, 4);
    
    // All 4 nodes should be assigned IDs
    EXPECT_EQ(new_to_orig.size(), 4);
    
    // Check that all mappings are valid
    for (const auto& id : new_to_orig) {
        EXPECT_LT(id, 4);
    }
}

TEST_F(GraphPreprocessorTest, RelabelEdges) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {1, 2}
    };
    
    // Simple relabeling: swap 0<->2
    std::vector<size_t> orig_to_new = {2, 1, 0, SIZE_MAX};
    
    auto relabeled = GraphPreprocessor::relabel_edges(edges, orig_to_new);
    
    EXPECT_EQ(relabeled.size(), 2);
    EXPECT_EQ(relabeled[0], std::make_pair(static_cast<size_t>(2), static_cast<size_t>(1)));
    EXPECT_EQ(relabeled[1], std::make_pair(static_cast<size_t>(1), static_cast<size_t>(0)));
}

TEST_F(GraphPreprocessorTest, RelabelEdgesSkipsInvalid) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {1, 5}  // Node 5 doesn't exist in mapping
    };
    
    std::vector<size_t> orig_to_new = {0, 1, 2};  // Only 3 nodes
    
    auto relabeled = GraphPreprocessor::relabel_edges(edges, orig_to_new);
    
    // Only valid edges should be kept
    EXPECT_EQ(relabeled.size(), 1);
    EXPECT_EQ(relabeled[0], std::make_pair(static_cast<size_t>(0), static_cast<size_t>(1)));
}

// ===== GEF_Graph Tests =====

TEST_F(GEFGraphTest, ConstructFromEdges) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 2}, {1, 2}, {2, 3}
    };
    
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(edges, 4, bv_factory);
    
    EXPECT_EQ(graph.num_nodes(), 4);
    EXPECT_EQ(graph.num_edges(), 4);
}

TEST_F(GEFGraphTest, OutDegree) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 2}, {0, 3},  // Node 0 has out-degree 3
        {1, 2},                   // Node 1 has out-degree 1
                                  // Node 2, 3 have out-degree 0
    };
    
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(edges, 4, bv_factory);
    
    EXPECT_EQ(graph.out_degree(0), 3);
    EXPECT_EQ(graph.out_degree(1), 1);
    EXPECT_EQ(graph.out_degree(2), 0);
    EXPECT_EQ(graph.out_degree(3), 0);
}

TEST_F(GEFGraphTest, InDegree) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 2}, {1, 2}, {3, 2},  // Node 2 has in-degree 3
        {0, 1},                   // Node 1 has in-degree 1
                                  // Node 0, 3 have in-degree 0
    };
    
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(edges, 4, bv_factory);
    
    EXPECT_EQ(graph.in_degree(0), 0);
    EXPECT_EQ(graph.in_degree(1), 1);
    EXPECT_EQ(graph.in_degree(2), 3);
    EXPECT_EQ(graph.in_degree(3), 0);
}

TEST_F(GEFGraphTest, OutNeighbors) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 3}, {0, 2}  // Neighbors of 0 should be sorted: 1, 2, 3
    };
    
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(edges, 4, bv_factory);
    
    auto neighbors = graph.out_neighbors(0);
    ASSERT_EQ(neighbors.size(), 3);
    
    // Neighbors should be sorted
    EXPECT_EQ(neighbors[0], 1);
    EXPECT_EQ(neighbors[1], 2);
    EXPECT_EQ(neighbors[2], 3);
}

TEST_F(GEFGraphTest, InNeighbors) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {1, 0}, {3, 0}, {2, 0}  // In-neighbors of 0 should be sorted: 1, 2, 3
    };
    
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(edges, 4, bv_factory);
    
    auto neighbors = graph.in_neighbors(0);
    ASSERT_EQ(neighbors.size(), 3);
    
    // Neighbors should be sorted
    EXPECT_EQ(neighbors[0], 1);
    EXPECT_EQ(neighbors[1], 2);
    EXPECT_EQ(neighbors[2], 3);
}

TEST_F(GEFGraphTest, RandomAccessNeighbor) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 10}, {0, 20}, {0, 30}
    };
    
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(edges, 31, bv_factory);
    
    EXPECT_EQ(graph.out_neighbor(0, 0), 10);
    EXPECT_EQ(graph.out_neighbor(0, 1), 20);
    EXPECT_EQ(graph.out_neighbor(0, 2), 30);
}

TEST_F(GEFGraphTest, EmptyGraph) {
    std::vector<std::pair<size_t, size_t>> edges;
    
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(edges, 0, bv_factory);
    
    EXPECT_EQ(graph.num_nodes(), 0);
    EXPECT_EQ(graph.num_edges(), 0);
}

TEST_F(GEFGraphTest, SizeInBytes) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 2}, {1, 2}, {2, 3}
    };
    
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(edges, 4, bv_factory);
    
    // Size should be positive
    EXPECT_GT(graph.size_in_bytes(), 0);
}

TEST_F(GEFGraphTest, WorksWithMyEF) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 2}, {1, 2}, {2, 3}
    };
    
    GEF_Graph<MyEF<uint32_t, PastaBitVector>> graph(edges, 4, bv_factory);
    
    EXPECT_EQ(graph.num_nodes(), 4);
    EXPECT_EQ(graph.num_edges(), 4);
    EXPECT_EQ(graph.out_degree(0), 2);
}

TEST_F(GEFGraphTest, PreprocessedGraphConstruction) {
    // Create a preprocessed graph manually
    PreprocessedGraph pg;
    pg.edges = {{0, 1}, {1, 2}};
    pg.num_nodes = 3;
    pg.num_edges = 2;
    pg.new_to_original = {10, 20, 30};  // Original IDs were 10, 20, 30
    pg.original_to_new.resize(31, SIZE_MAX);
    pg.original_to_new[10] = 0;
    pg.original_to_new[20] = 1;
    pg.original_to_new[30] = 2;
    
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(pg, bv_factory);
    
    EXPECT_EQ(graph.num_nodes(), 3);
    EXPECT_TRUE(graph.has_original_mapping());
    EXPECT_EQ(graph.to_original_id(0), 10);
    EXPECT_EQ(graph.to_original_id(1), 20);
    EXPECT_EQ(graph.to_original_id(2), 30);
}

// Large graph test
TEST_F(GEFGraphTest, LargerGraph) {
    // Create a denser graph
    std::vector<std::pair<size_t, size_t>> edges;
    const size_t n = 100;
    
    // Add edges: each node i connects to (i+1) % n and (i+10) % n
    for (size_t i = 0; i < n; ++i) {
        edges.emplace_back(i, (i + 1) % n);
        edges.emplace_back(i, (i + 10) % n);
    }
    
    GEF_Graph<B_GEF<uint32_t, PastaBitVector>> graph(edges, n, bv_factory);
    
    EXPECT_EQ(graph.num_nodes(), n);
    EXPECT_EQ(graph.num_edges(), 2 * n);
    
    // Each node should have out-degree 2 and in-degree 2
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(graph.out_degree(i), 2);
        EXPECT_EQ(graph.in_degree(i), 2);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


