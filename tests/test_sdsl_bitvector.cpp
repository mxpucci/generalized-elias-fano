#include <gtest/gtest.h>
#include "../include/datastructures/SDSLBitVector.hpp"
#include "../include/datastructures/SDSLBitVectorFactory.hpp"
#include <filesystem>
#include <fstream>
#include <vector>

class SDSLBitVectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        factory = std::make_unique<SDSLBitVectorFactory>();
    }

    void TearDown() override {
        // Clean up any test files
        for (const auto& path : test_files) {
            if (std::filesystem::exists(path)) {
                std::filesystem::remove(path);
            }
        }
    }

    std::unique_ptr<SDSLBitVectorFactory> factory;
    std::vector<std::filesystem::path> test_files;
};

// Test constructors
TEST_F(SDSLBitVectorTest, ConstructorFromSize) {
    SDSLBitVector bv(100);
    EXPECT_EQ(bv.size(), 100);
    EXPECT_FALSE(bv.empty());
    
    // All bits should be initialized to 0
    for (size_t i = 0; i < 100; i++) {
        EXPECT_FALSE(bv[i]);
    }
}

TEST_F(SDSLBitVectorTest, ConstructorFromSizeZero) {
    SDSLBitVector bv(0);
    EXPECT_EQ(bv.size(), 0);
    EXPECT_TRUE(bv.empty());
}

TEST_F(SDSLBitVectorTest, ConstructorFromSDSLBitVector) {
    sdsl::bit_vector sdsl_bv(50, 1); // 50 bits, all set to 1
    SDSLBitVector bv(std::move(sdsl_bv));
    
    EXPECT_EQ(bv.size(), 50);
    for (size_t i = 0; i < 50; i++) {
        EXPECT_TRUE(bv[i]);
    }
}

TEST_F(SDSLBitVectorTest, ConstructorFromVectorBool) {
    std::vector<bool> bits = {true, false, true, false, true};
    SDSLBitVector bv(bits);
    
    EXPECT_EQ(bv.size(), 5);
    EXPECT_TRUE(bv[0]);
    EXPECT_FALSE(bv[1]);
    EXPECT_TRUE(bv[2]);
    EXPECT_FALSE(bv[3]);
    EXPECT_TRUE(bv[4]);
}

TEST_F(SDSLBitVectorTest, ConstructorFromEmptyVectorBool) {
    std::vector<bool> bits;
    SDSLBitVector bv(bits);
    
    EXPECT_EQ(bv.size(), 0);
    EXPECT_TRUE(bv.empty());
}

// Test bit access and modification
TEST_F(SDSLBitVectorTest, BitAccessAndModification) {
    SDSLBitVector bv(10);
    
    // Test non-const operator[]
    bv[0] = 1;
    bv[5] = 1;
    bv[9] = 1;
    
    // Test const operator[]
    const SDSLBitVector& const_bv = bv;
    EXPECT_TRUE(const_bv[0]);
    EXPECT_FALSE(const_bv[1]);
    EXPECT_TRUE(const_bv[5]);
    EXPECT_TRUE(const_bv[9]);
}

// Test Rule of 5 - Copy constructor
TEST_F(SDSLBitVectorTest, CopyConstructor) {
    SDSLBitVector original(20);
    original[5] = 1;
    original[10] = 1;
    original.enable_rank();
    original.enable_select1();
    original.enable_select0();
    
    SDSLBitVector copy(original);
    
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy[5], original[5]);
    EXPECT_EQ(copy[10], original[10]);
    
    // Verify rank/select work on copy
    EXPECT_EQ(copy.rank(20), 2);
    EXPECT_EQ(copy.select(1), 5);
    EXPECT_EQ(copy.select0(1), 0);
}

TEST_F(SDSLBitVectorTest, CopyConstructorWithoutSupport) {
    SDSLBitVector original(10);
    original[3] = 1;
    
    SDSLBitVector copy(original);
    
    EXPECT_EQ(copy.size(), 10);
    EXPECT_TRUE(copy[3]);
    
    // Should throw since no rank support
    EXPECT_THROW(copy.rank(5), std::runtime_error);
}

// Test Rule of 5 - Copy assignment
TEST_F(SDSLBitVectorTest, CopyAssignment) {
    SDSLBitVector original(15);
    original[7] = 1;
    original.enable_rank();
    original.enable_select1();
    
    SDSLBitVector copy(5); // Different size initially
    copy = original;
    
    EXPECT_EQ(copy.size(), 15);
    EXPECT_TRUE(copy[7]);
    EXPECT_EQ(copy.rank(15), 1);
}

TEST_F(SDSLBitVectorTest, CopyAssignmentSelfAssignment) {
    SDSLBitVector bv(10);
    bv[3] = 1;
    bv.enable_rank();
    
    bv = bv; // Self assignment
    
    EXPECT_EQ(bv.size(), 10);
    EXPECT_TRUE(bv[3]);
    EXPECT_EQ(bv.rank(10), 1);
}

// Test Rule of 5 - Move constructor
TEST_F(SDSLBitVectorTest, MoveConstructor) {
    SDSLBitVector original(25);
    original[12] = 1;
    original.enable_rank();
    original.enable_select0();
    
    size_t original_size = original.size();
    SDSLBitVector moved(std::move(original));
    
    EXPECT_EQ(moved.size(), original_size);
    EXPECT_TRUE(moved[12]);
    EXPECT_EQ(moved.rank(25), 1);
    EXPECT_EQ(moved.select0(1), 0);
}

// Test Rule of 5 - Move assignment
TEST_F(SDSLBitVectorTest, MoveAssignment) {
    SDSLBitVector original(30);
    original[15] = 1;
    original.enable_rank();
    
    SDSLBitVector moved(5);
    moved = std::move(original);
    
    EXPECT_EQ(moved.size(), 30);
    EXPECT_TRUE(moved[15]);
    EXPECT_EQ(moved.rank(30), 1);
}

TEST_F(SDSLBitVectorTest, MoveAssignmentSelfAssignment) {
    SDSLBitVector bv(10);
    bv[3] = 1;
    bv.enable_rank();
    
    bv = std::move(bv); // Self assignment
    
    EXPECT_EQ(bv.size(), 10);
    EXPECT_TRUE(bv[3]);
    EXPECT_EQ(bv.rank(10), 1);
}

// Test rank operations
TEST_F(SDSLBitVectorTest, RankOperations) {
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1}; // 6 ones
    SDSLBitVector bv(bits);
    bv.enable_rank();
    
    EXPECT_EQ(bv.rank(0), 0);
    EXPECT_EQ(bv.rank(1), 1);
    EXPECT_EQ(bv.rank(3), 2);
    EXPECT_EQ(bv.rank(6), 4);
    EXPECT_EQ(bv.rank(10), 6);
    
    // Test inherited rank(start, end) method
    EXPECT_EQ(bv.rank(2, 6), 3); // positions 2,3,5 have ones
}

TEST_F(SDSLBitVectorTest, RankWithoutSupport) {
    SDSLBitVector bv(10);
    EXPECT_THROW(bv.rank(5), std::runtime_error);
}

TEST_F(SDSLBitVectorTest, RankInheritedMethods) {
    std::vector<bool> bits = {1, 0, 1, 1, 0};
    SDSLBitVector bv(bits);
    bv.enable_rank();
    
    // Test rank0 methods
    EXPECT_EQ(bv.rank0(5), 2); // 2 zeros in first 5 positions
    EXPECT_EQ(bv.rank0(1, 4), 1); // 1 zero in positions 1-3
    
    // Test range rank with invalid range
    EXPECT_THROW(bv.rank(3, 2), std::out_of_range);
}

// Test select operations
TEST_F(SDSLBitVectorTest, SelectOperations) {
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1}; // positions of 1s: 0,2,3,5,8,9
    SDSLBitVector bv(bits);
    bv.enable_select1();
    
    EXPECT_EQ(bv.select(1), 0);
    EXPECT_EQ(bv.select(2), 2);
    EXPECT_EQ(bv.select(3), 3);
    EXPECT_EQ(bv.select(4), 5);
    EXPECT_EQ(bv.select(5), 8);
    EXPECT_EQ(bv.select(6), 9);
}

TEST_F(SDSLBitVectorTest, Select0Operations) {
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1}; // positions of 0s: 1,4,6,7
    SDSLBitVector bv(bits);
    bv.enable_select0();
    
    EXPECT_EQ(bv.select0(1), 1);
    EXPECT_EQ(bv.select0(2), 4);
    EXPECT_EQ(bv.select0(3), 6);
    EXPECT_EQ(bv.select0(4), 7);
}

TEST_F(SDSLBitVectorTest, SelectWithoutSupport) {
    SDSLBitVector bv(10);
    bv[5] = 1;
    
    EXPECT_THROW(bv.select(1), std::runtime_error);
    EXPECT_THROW(bv.select0(1), std::runtime_error);
}

// Test enable methods
TEST_F(SDSLBitVectorTest, EnableMethods) {
    SDSLBitVector bv(10);
    bv[5] = 1;
    
    // Initially should throw
    EXPECT_THROW(bv.rank(10), std::runtime_error);
    EXPECT_THROW(bv.select(1), std::runtime_error);
    EXPECT_THROW(bv.select0(1), std::runtime_error);
    
    // Enable rank
    bv.enable_rank();
    EXPECT_EQ(bv.rank(10), 1);
    
    // Enable select1
    bv.enable_select1();
    EXPECT_EQ(bv.select(1), 5);
    
    // Enable select0
    bv.enable_select0();
    EXPECT_EQ(bv.select0(1), 0);
    
    // Calling enable again should be safe
    bv.enable_rank();
    bv.enable_select1();
    bv.enable_select0();
    
    EXPECT_EQ(bv.rank(10), 1);
    EXPECT_EQ(bv.select(1), 5);
    EXPECT_EQ(bv.select0(1), 0);
}

// Test memory operations
TEST_F(SDSLBitVectorTest, MemoryOperations) {
    SDSLBitVector bv(1000);
    
    size_t base_size = bv.size_in_bytes();
    EXPECT_GT(base_size, 0);
    
    bv.enable_rank();
    size_t with_rank = bv.size_in_bytes();
    EXPECT_GT(with_rank, base_size);
    
    bv.enable_select1();
    size_t with_select1 = bv.size_in_bytes();
    EXPECT_GT(with_select1, with_rank);
    
    bv.enable_select0();
    size_t with_select0 = bv.size_in_bytes();
    EXPECT_GT(with_select0, with_select1);
    
    // Test megabytes calculation
    size_t mb = bv.size_in_megabytes();
    EXPECT_EQ(mb, (with_select0 + 1024 * 1024 - 1) / (1024 * 1024));
}

// Test serialization and loading
TEST_F(SDSLBitVectorTest, SerializationAndLoading) {
    std::filesystem::path test_file = "test_bitvector.bin";
    test_files.push_back(test_file);
    
    // Create and serialize
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    SDSLBitVector original(bits);
    original.serialize(test_file);
    
    // Load and verify
    SDSLBitVector loaded = SDSLBitVector::load(test_file);
    EXPECT_EQ(loaded.size(), original.size());
    
    for (size_t i = 0; i < loaded.size(); i++) {
        EXPECT_EQ(loaded[i], original[i]);
    }
}

TEST_F(SDSLBitVectorTest, SerializationError) {
    SDSLBitVector bv(10);
    
    // Try to serialize to invalid path
    EXPECT_THROW(bv.serialize("/invalid/path/file.bin"), std::runtime_error);
}

TEST_F(SDSLBitVectorTest, LoadError) {
    // Try to load from non-existent file
    EXPECT_THROW(SDSLBitVector::load("non_existent_file.bin"), std::runtime_error);
}

// Factory tests
class SDSLBitVectorFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        factory = std::make_unique<SDSLBitVectorFactory>();
    }

    void TearDown() override {
        for (const auto& path : test_files) {
            if (std::filesystem::exists(path)) {
                std::filesystem::remove(path);
            }
        }
    }

    std::unique_ptr<SDSLBitVectorFactory> factory;
    std::vector<std::filesystem::path> test_files;
};

TEST_F(SDSLBitVectorFactoryTest, CreateFromSize) {
    auto bv = factory->create(100);
    
    EXPECT_EQ(bv->size(), 100);
    EXPECT_FALSE(bv->empty());
    
    for (size_t i = 0; i < 100; i++) {
        EXPECT_FALSE((*bv)[i]);
    }
}

TEST_F(SDSLBitVectorFactoryTest, CreateFromSizeZero) {
    auto bv = factory->create(0);
    
    EXPECT_EQ(bv->size(), 0);
    EXPECT_TRUE(bv->empty());
}

TEST_F(SDSLBitVectorFactoryTest, CreateFromVectorBool) {
    std::vector<bool> bits = {true, false, true, false, true};
    auto bv = factory->create(bits);
    
    EXPECT_EQ(bv->size(), 5);
    EXPECT_TRUE((*bv)[0]);
    EXPECT_FALSE((*bv)[1]);
    EXPECT_TRUE((*bv)[2]);
    EXPECT_FALSE((*bv)[3]);
    EXPECT_TRUE((*bv)[4]);
}

TEST_F(SDSLBitVectorFactoryTest, CreateFromEmptyVectorBool) {
    std::vector<bool> bits;
    auto bv = factory->create(bits);
    
    EXPECT_EQ(bv->size(), 0);
    EXPECT_TRUE(bv->empty());
}

TEST_F(SDSLBitVectorFactoryTest, FromFile) {
    std::filesystem::path test_file = "test_factory_file.bin";
    test_files.push_back(test_file);
    
    // Create a test file
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1};
    SDSLBitVector original(bits);
    original.serialize(test_file);
    
    // Load via factory
    auto loaded = factory->from_file(test_file);
    
    EXPECT_EQ(loaded->size(), 6);
    EXPECT_TRUE((*loaded)[0]);
    EXPECT_FALSE((*loaded)[1]);
    EXPECT_TRUE((*loaded)[2]);
    EXPECT_TRUE((*loaded)[3]);
    EXPECT_FALSE((*loaded)[4]);
    EXPECT_TRUE((*loaded)[5]);
}

TEST_F(SDSLBitVectorFactoryTest, FromFileError) {
    // Try to load from non-existent file
    EXPECT_THROW(factory->from_file("non_existent_file.bin"), std::runtime_error);
}

// Integration test: verify factory creates SDSLBitVector instances
TEST_F(SDSLBitVectorFactoryTest, FactoryCreatesSdslBitVector) {
    auto bv = factory->create(10);
    
    // Enable features to test they work
    bv->enable_rank();
    bv->enable_select1();
    bv->enable_select0();
    
    // This should not throw
    EXPECT_NO_THROW(bv->rank(10));
}

// Edge case tests
TEST_F(SDSLBitVectorTest, LargeBitVector) {
    SDSLBitVector bv(1000000);
    EXPECT_EQ(bv.size(), 1000000);
    
    bv[999999] = 1;
    EXPECT_TRUE(bv[999999]);
    
    bv.enable_rank();
    EXPECT_EQ(bv.rank(1000000), 1);
}

TEST_F(SDSLBitVectorTest, AllOnesVector) {
    std::vector<bool> all_ones(100, true);
    SDSLBitVector bv(all_ones);
    bv.enable_rank();
    bv.enable_select1();
    bv.enable_select0();
    
    EXPECT_EQ(bv.rank(100), 100);
    EXPECT_EQ(bv.select(50), 49); // 50th one is at position 49
    
    // No zeros, so check total zeros via rank0
    EXPECT_EQ(bv.rank0(100), 0);
}

TEST_F(SDSLBitVectorTest, AllZerosVector) {
    std::vector<bool> all_zeros(100, false);
    SDSLBitVector bv(all_zeros);
    bv.enable_rank();
    bv.enable_select1();
    bv.enable_select0();
    
    EXPECT_EQ(bv.rank(100), 0);
    EXPECT_EQ(bv.select0(50), 49); // 50th zero is at position 49
    // Check total ones via rank
    EXPECT_EQ(bv.rank(100), 0);
} 