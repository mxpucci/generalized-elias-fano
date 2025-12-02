//
// Tests for GEF_Graph, GraphPreprocessor, and U_GEF_AdjList
//

#include <gtest/gtest.h>
#include "../include/graphs/GEF_Graph.hpp"
#include "../include/graphs/GraphPreprocessor.hpp"
#include "../include/graphs/GEF_AdjList.hpp"
#include "../include/graphs/U_GEF_AdjList.hpp"
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

// ===== U_GEF_AdjList Tests =====

class UGEFAdjListTest : public ::testing::Test {
protected:
    std::shared_ptr<IBitVectorFactory> bv_factory;
    
    void SetUp() override {
        bv_factory = std::make_shared<PastaBitVectorFactory>();
    }
    void TearDown() override {}
};

TEST_F(UGEFAdjListTest, BasicConstruction) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 2}, {1, 2}, {2, 3}
    };
    
    U_GEF_AdjList<uint32_t> adj(edges, 4, 4, false, bv_factory);
    
    EXPECT_EQ(adj.num_nodes(), 4);
    EXPECT_EQ(adj.num_edges(), 4);
}

TEST_F(UGEFAdjListTest, DegreeQuery) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 2}, {0, 3},  // Node 0 has degree 3
        {1, 2},                   // Node 1 has degree 1
                                  // Node 2, 3 have degree 0
    };
    
    U_GEF_AdjList<uint32_t> adj(edges, 4, 4, false, bv_factory);
    
    EXPECT_EQ(adj.degree(0), 3);
    EXPECT_EQ(adj.degree(1), 1);
    EXPECT_EQ(adj.degree(2), 0);
    EXPECT_EQ(adj.degree(3), 0);
}

TEST_F(UGEFAdjListTest, GetIthNeighbor) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 10}, {0, 20}, {0, 30}
    };
    
    U_GEF_AdjList<uint32_t> adj(edges, 31, 3, false, bv_factory);
    
    EXPECT_EQ(adj.getIthNeighbor(0, 0), 10);
    EXPECT_EQ(adj.getIthNeighbor(0, 1), 20);
    EXPECT_EQ(adj.getIthNeighbor(0, 2), 30);
}

TEST_F(UGEFAdjListTest, GetNeighbors) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 3}, {0, 2}  // Neighbors should be sorted: 1, 2, 3
    };
    
    U_GEF_AdjList<uint32_t> adj(edges, 4, 3, false, bv_factory);
    
    auto neighbors = adj.getNeighbors(0);
    ASSERT_EQ(neighbors.size(), 3);
    
    // Neighbors should be sorted
    EXPECT_EQ(neighbors[0], 1);
    EXPECT_EQ(neighbors[1], 2);
    EXPECT_EQ(neighbors[2], 3);
}

TEST_F(UGEFAdjListTest, EmptyNeighborhood) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {2, 3}  // Node 1 has no outgoing edges
    };
    
    U_GEF_AdjList<uint32_t> adj(edges, 4, 2, false, bv_factory);
    
    EXPECT_EQ(adj.degree(1), 0);
    auto neighbors = adj.getNeighbors(1);
    EXPECT_TRUE(neighbors.empty());
}

TEST_F(UGEFAdjListTest, EmptyGraph) {
    std::vector<std::pair<size_t, size_t>> edges;
    
    U_GEF_AdjList<uint32_t> adj(edges, 0, 0, false, bv_factory);
    
    EXPECT_EQ(adj.num_nodes(), 0);
    EXPECT_EQ(adj.num_edges(), 0);
}

TEST_F(UGEFAdjListTest, GraphWithOnlyEmptyNodes) {
    std::vector<std::pair<size_t, size_t>> edges;
    
    U_GEF_AdjList<uint32_t> adj(edges, 5, 0, false, bv_factory);
    
    EXPECT_EQ(adj.num_nodes(), 5);
    EXPECT_EQ(adj.num_edges(), 0);
    
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(adj.degree(i), 0);
    }
}

TEST_F(UGEFAdjListTest, ReverseEdges) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 2}, {0, 3}
    };
    
    U_GEF_AdjList<uint32_t> adj(edges, 4, 3, true, bv_factory);  // reverse = true
    
    // With reverse, the edges become 1->0, 2->0, 3->0
    EXPECT_EQ(adj.degree(0), 0);  // Node 0 has no outgoing edges
    EXPECT_EQ(adj.degree(1), 1);
    EXPECT_EQ(adj.degree(2), 1);
    EXPECT_EQ(adj.degree(3), 1);
    
    EXPECT_EQ(adj.getIthNeighbor(1, 0), 0);
    EXPECT_EQ(adj.getIthNeighbor(2, 0), 0);
    EXPECT_EQ(adj.getIthNeighbor(3, 0), 0);
}

TEST_F(UGEFAdjListTest, DuplicateEdges) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 1}, {0, 2}, {0, 2}, {0, 2}  // Duplicates should be removed
    };
    
    U_GEF_AdjList<uint32_t> adj(edges, 3, 5, false, bv_factory);
    
    EXPECT_EQ(adj.degree(0), 2);  // Only 2 unique neighbors: 1, 2
    EXPECT_EQ(adj.num_edges(), 2);
    
    auto neighbors = adj.getNeighbors(0);
    ASSERT_EQ(neighbors.size(), 2);
    EXPECT_EQ(neighbors[0], 1);
    EXPECT_EQ(neighbors[1], 2);
}

TEST_F(UGEFAdjListTest, LargerGraph) {
    std::vector<std::pair<size_t, size_t>> edges;
    const size_t n = 100;
    
    // Each node i connects to (i+1) % n and (i+10) % n
    for (size_t i = 0; i < n; ++i) {
        edges.emplace_back(i, (i + 1) % n);
        edges.emplace_back(i, (i + 10) % n);
    }
    
    U_GEF_AdjList<uint32_t> adj(edges, n, 2 * n, false, bv_factory);
    
    EXPECT_EQ(adj.num_nodes(), n);
    EXPECT_EQ(adj.num_edges(), 2 * n);
    
    // Each node should have degree 2
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(adj.degree(i), 2);
    }
    
    // Verify specific neighbors
    for (size_t i = 0; i < n; ++i) {
        auto neighbors = adj.getNeighbors(i);
        ASSERT_EQ(neighbors.size(), 2);
        
        // Neighbors should be sorted
        size_t n1 = (i + 1) % n;
        size_t n2 = (i + 10) % n;
        if (n1 > n2) std::swap(n1, n2);
        
        EXPECT_EQ(neighbors[0], n1);
        EXPECT_EQ(neighbors[1], n2);
    }
}

TEST_F(UGEFAdjListTest, SizeInBytes) {
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 1}, {0, 2}, {1, 2}, {2, 3}
    };
    
    U_GEF_AdjList<uint32_t> adj(edges, 4, 4, false, bv_factory);
    
    // Size should be positive
    EXPECT_GT(adj.size_in_bytes(), 0);
}

TEST_F(UGEFAdjListTest, CompareWithGEFAdjList) {
    // Verify that U_GEF_AdjList produces the same results as GEF_AdjList
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 5}, {0, 10}, {0, 15},
        {1, 3}, {1, 7},
        {2, 1}, {2, 4}, {2, 8}, {2, 9},
        {3, 0}
    };
    
    GEF_AdjList<B_GEF<uint32_t, PastaBitVector>> gef_adj(edges, 16, 10, false, bv_factory);
    U_GEF_AdjList<uint32_t> u_gef_adj(edges, 16, 10, false, bv_factory);
    
    EXPECT_EQ(gef_adj.num_nodes(), u_gef_adj.num_nodes());
    EXPECT_EQ(gef_adj.num_edges(), u_gef_adj.num_edges());
    
    // Compare all neighborhoods
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(gef_adj.degree(i), u_gef_adj.degree(i)) << "Degree mismatch at node " << i;
        
        auto gef_neighbors = gef_adj.getNeighbors(i);
        auto u_gef_neighbors = u_gef_adj.getNeighbors(i);
        
        ASSERT_EQ(gef_neighbors.size(), u_gef_neighbors.size()) << "Size mismatch at node " << i;
        
        for (size_t j = 0; j < gef_neighbors.size(); ++j) {
            EXPECT_EQ(gef_neighbors[j], u_gef_neighbors[j]) 
                << "Neighbor mismatch at node " << i << ", index " << j;
        }
    }
}

TEST_F(UGEFAdjListTest, SingleDegreeNodes) {
    // Graph where every node has exactly degree 1
    std::vector<std::pair<size_t, size_t>> edges;
    const size_t n = 50;
    
    for (size_t i = 0; i < n; ++i) {
        edges.emplace_back(i, (i + 1) % n);
    }
    
    U_GEF_AdjList<uint32_t> adj(edges, n, n, false, bv_factory);
    
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(adj.degree(i), 1);
        EXPECT_EQ(adj.getIthNeighbor(i, 0), (i + 1) % n);
    }
}

TEST_F(UGEFAdjListTest, HighDegreeNode) {
    // One node with very high degree
    std::vector<std::pair<size_t, size_t>> edges;
    const size_t n = 1000;
    
    for (size_t i = 1; i < n; ++i) {
        edges.emplace_back(0, i);
    }
    
    U_GEF_AdjList<uint32_t> adj(edges, n, n - 1, false, bv_factory);
    
    EXPECT_EQ(adj.degree(0), n - 1);
    
    auto neighbors = adj.getNeighbors(0);
    ASSERT_EQ(neighbors.size(), n - 1);
    
    // Verify neighbors are sorted 1, 2, 3, ..., n-1
    for (size_t i = 0; i < n - 1; ++i) {
        EXPECT_EQ(neighbors[i], i + 1);
    }
    
    // Verify random access
    for (size_t i = 0; i < n - 1; ++i) {
        EXPECT_EQ(adj.getIthNeighbor(0, i), i + 1);
    }
}

TEST_F(UGEFAdjListTest, SerializationDeserialization) {
    // Create a graph
    std::vector<std::pair<size_t, size_t>> edges = {
        {0, 5}, {0, 10}, {0, 15},
        {1, 3}, {1, 7},
        {2, 1}, {2, 4}, {2, 8}, {2, 9},
        {3, 0}
    };
    
    U_GEF_AdjList<uint32_t> original(edges, 16, 10, false, bv_factory);
    
    // Serialize
    std::string temp_file = "/tmp/u_gef_adjlist_test.bin";
    {
        std::ofstream ofs(temp_file, std::ios::binary);
        original.serialize(ofs);
    }
    
    // Deserialize
    U_GEF_AdjList<uint32_t> loaded;
    {
        std::ifstream ifs(temp_file, std::ios::binary);
        loaded.load(ifs, bv_factory);
    }
    
    // Verify
    EXPECT_EQ(loaded.num_nodes(), original.num_nodes());
    EXPECT_EQ(loaded.num_edges(), original.num_edges());
    
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(loaded.degree(i), original.degree(i));
        
        auto orig_neighbors = original.getNeighbors(i);
        auto load_neighbors = loaded.getNeighbors(i);
        
        ASSERT_EQ(orig_neighbors.size(), load_neighbors.size());
        
        for (size_t j = 0; j < orig_neighbors.size(); ++j) {
            EXPECT_EQ(orig_neighbors[j], load_neighbors[j]);
        }
    }
    
    // Clean up
    std::remove(temp_file.c_str());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}





