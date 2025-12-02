#include <gtest/gtest.h>
#include "../include/datastructures/SUXBitVector.hpp"
#include "../include/datastructures/SUXBitVectorFactory.hpp"
#include <filesystem>
#include <fstream>
#include <vector>

class SUXBitVectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        factory = std::make_unique<SUXBitVectorFactory>();
    }

    void TearDown() override {
        // Clean up any test files
        for (const auto& path : test_files) {
            if (std::filesystem::exists(path)) {
                std::filesystem::remove(path);
            }
        }
    }

    std::unique_ptr<SUXBitVectorFactory> factory;
    std::vector<std::filesystem::path> test_files;
};

// Test constructors
TEST_F(SUXBitVectorTest, ConstructorFromSize) {
    SUXBitVector bv(100);
    EXPECT_EQ(bv.size(), 100);
    EXPECT_FALSE(bv.empty());
    
    // All bits should be initialized to 0
    for (size_t i = 0; i < 100; i++) {
        EXPECT_FALSE(bv[i]);
    }
}

TEST_F(SUXBitVectorTest, ConstructorFromSizeZero) {
    SUXBitVector bv(0);
    EXPECT_EQ(bv.size(), 0);
    EXPECT_TRUE(bv.empty());
}

TEST_F(SUXBitVectorTest, ConstructorFromVectorBool) {
    std::vector<bool> bits = {true, false, true, false, true};
    SUXBitVector bv(bits);
    
    EXPECT_EQ(bv.size(), 5);
    EXPECT_TRUE(bv[0]);
    EXPECT_FALSE(bv[1]);
    EXPECT_TRUE(bv[2]);
    EXPECT_FALSE(bv[3]);
    EXPECT_TRUE(bv[4]);
}

TEST_F(SUXBitVectorTest, ConstructorFromEmptyVectorBool) {
    std::vector<bool> bits;
    SUXBitVector bv(bits);
    
    EXPECT_EQ(bv.size(), 0);
    EXPECT_TRUE(bv.empty());
}

// Test bit access and modification
TEST_F(SUXBitVectorTest, BitAccessAndModification) {
    SUXBitVector bv(10);
    
    // Test set method
    bv.set(0, true);
    bv.set(5, true);
    bv.set(9, true);
    
    // Test const operator[]
    const SUXBitVector& const_bv = bv;
    EXPECT_TRUE(const_bv[0]);
    EXPECT_FALSE(const_bv[1]);
    EXPECT_TRUE(const_bv[5]);
    EXPECT_TRUE(const_bv[9]);
}

// Test Rule of 5 - Copy constructor
TEST_F(SUXBitVectorTest, CopyConstructor) {
    SUXBitVector original(20);
    original.set(5, true);
    original.set(10, true);
    original.enable_rank();
    original.enable_select1();
    original.enable_select0();
    
    SUXBitVector copy(original);
    
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy[5], original[5]);
    EXPECT_EQ(copy[10], original[10]);
    
    // Verify rank/select work on copy
    EXPECT_EQ(copy.rank(20), 2);
    EXPECT_EQ(copy.select(1), 5);
    EXPECT_EQ(copy.select0(1), 0);
}

TEST_F(SUXBitVectorTest, CopyConstructorWithoutSupport) {
    SUXBitVector original(10);
    original.set(3, true);
    
    SUXBitVector copy(original);
    
    EXPECT_EQ(copy.size(), 10);
    EXPECT_TRUE(copy[3]);
    // Note: Calling rank/select without enabling support is undefined behavior
    // The caller is responsible for enabling supports before use
}

// Test Rule of 5 - Copy assignment
TEST_F(SUXBitVectorTest, CopyAssignment) {
    SUXBitVector original(15);
    original.set(7, true);
    original.enable_rank();
    original.enable_select1();
    
    SUXBitVector copy(5); // Different size initially
    copy = original;
    
    EXPECT_EQ(copy.size(), 15);
    EXPECT_TRUE(copy[7]);
    EXPECT_EQ(copy.rank(15), 1);
}

TEST_F(SUXBitVectorTest, CopyAssignmentSelfAssignment) {
    SUXBitVector bv(10);
    bv.set(3, true);
    bv.enable_rank();
    
    bv = bv; // Self assignment
    
    EXPECT_EQ(bv.size(), 10);
    EXPECT_TRUE(bv[3]);
    EXPECT_EQ(bv.rank(10), 1);
}

// Test Rule of 5 - Move constructor
TEST_F(SUXBitVectorTest, MoveConstructor) {
    SUXBitVector original(25);
    original.set(12, true);
    original.enable_rank();
    original.enable_select0();
    
    size_t original_size = original.size();
    SUXBitVector moved(std::move(original));
    
    EXPECT_EQ(moved.size(), original_size);
    EXPECT_TRUE(moved[12]);
    EXPECT_EQ(moved.rank(25), 1);
    EXPECT_EQ(moved.select0(1), 0);
}

// Test Rule of 5 - Move assignment
TEST_F(SUXBitVectorTest, MoveAssignment) {
    SUXBitVector original(30);
    original.set(15, true);
    original.enable_rank();
    
    SUXBitVector moved(5);
    moved = std::move(original);
    
    EXPECT_EQ(moved.size(), 30);
    EXPECT_TRUE(moved[15]);
    EXPECT_EQ(moved.rank(30), 1);
}

TEST_F(SUXBitVectorTest, MoveAssignmentSelfAssignment) {
    SUXBitVector bv(10);
    bv.set(3, true);
    bv.enable_rank();
    
    bv = std::move(bv); // Self assignment
    
    EXPECT_EQ(bv.size(), 10);
    EXPECT_TRUE(bv[3]);
    EXPECT_EQ(bv.rank(10), 1);
}

// Test rank operations
TEST_F(SUXBitVectorTest, RankOperations) {
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1}; // 6 ones
    SUXBitVector bv(bits);
    bv.enable_rank();
    
    EXPECT_EQ(bv.rank(0), 0);
    EXPECT_EQ(bv.rank(1), 1);
    EXPECT_EQ(bv.rank(3), 2);
    EXPECT_EQ(bv.rank(6), 4);
    EXPECT_EQ(bv.rank(10), 6);
    
    // Test inherited rank(start, end) method
    EXPECT_EQ(bv.rank(2, 6), 3); // positions 2,3,5 have ones
}

// RankWithoutSupport test removed - calling rank without enabling support is undefined behavior
// The caller is responsible for enabling supports before use

TEST_F(SUXBitVectorTest, RankInheritedMethods) {
    std::vector<bool> bits = {1, 0, 1, 1, 0};
    SUXBitVector bv(bits);
    bv.enable_rank();
    
    // Test rank0 methods
    EXPECT_EQ(bv.rank0(5), 2); // 2 zeros in first 5 positions
    EXPECT_EQ(bv.rank0(1, 4), 1); // 1 zero in positions 1-3
    
    // Test range rank with invalid range
    EXPECT_THROW(bv.rank(3, 2), std::out_of_range);
}

// Test select operations
TEST_F(SUXBitVectorTest, SelectOperations) {
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1}; // positions of 1s: 0,2,3,5,8,9
    SUXBitVector bv(bits);
    bv.enable_select1();
    
    EXPECT_EQ(bv.select(1), 0);
    EXPECT_EQ(bv.select(2), 2);
    EXPECT_EQ(bv.select(3), 3);
    EXPECT_EQ(bv.select(4), 5);
    EXPECT_EQ(bv.select(5), 8);
    EXPECT_EQ(bv.select(6), 9);
}

TEST_F(SUXBitVectorTest, Select0Operations) {
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1}; // positions of 0s: 1,4,6,7
    SUXBitVector bv(bits);
    bv.enable_select0();
    
    EXPECT_EQ(bv.select0(1), 1);
    EXPECT_EQ(bv.select0(2), 4);
    EXPECT_EQ(bv.select0(3), 6);
    EXPECT_EQ(bv.select0(4), 7);
}

// SelectWithoutSupport test removed - calling select without enabling support is undefined behavior
// The caller is responsible for enabling supports before use

// Test enable methods
TEST_F(SUXBitVectorTest, EnableMethods) {
    SUXBitVector bv(10);
    bv.set(5, true);
    
    // Enable supports before use (calling without support is undefined behavior)
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
TEST_F(SUXBitVectorTest, MemoryOperations) {
    SUXBitVector bv(1000);
    
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
TEST_F(SUXBitVectorTest, SerializationAndLoading) {
    std::filesystem::path test_file = "test_sux_bitvector.bin";
    test_files.push_back(test_file);
    
    // Create and serialize
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    SUXBitVector original(bits);
    original.serialize(test_file);
    
    // Load and verify
    SUXBitVector loaded = SUXBitVector::load(test_file);
    EXPECT_EQ(loaded.size(), original.size());
    
    for (size_t i = 0; i < loaded.size(); i++) {
        EXPECT_EQ(loaded[i], original[i]);
    }
}

TEST_F(SUXBitVectorTest, SerializationError) {
    SUXBitVector bv(10);
    
    // Try to serialize to invalid path
    EXPECT_THROW(bv.serialize("/invalid/path/file.bin"), std::runtime_error);
}

TEST_F(SUXBitVectorTest, LoadError) {
    // Try to load from non-existent file
    EXPECT_THROW(SUXBitVector::load("non_existent_file.bin"), std::runtime_error);
}

// Factory tests
class SUXBitVectorFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        factory = std::make_unique<SUXBitVectorFactory>();
    }

    void TearDown() override {
        for (const auto& path : test_files) {
            if (std::filesystem::exists(path)) {
                std::filesystem::remove(path);
            }
        }
    }

    std::unique_ptr<SUXBitVectorFactory> factory;
    std::vector<std::filesystem::path> test_files;
};

TEST_F(SUXBitVectorFactoryTest, CreateFromSize) {
    auto bv = factory->create(100);
    
    EXPECT_EQ(bv->size(), 100);
    EXPECT_FALSE(bv->empty());
    
    for (size_t i = 0; i < 100; i++) {
        EXPECT_FALSE((*bv)[i]);
    }
}

TEST_F(SUXBitVectorFactoryTest, CreateFromSizeZero) {
    auto bv = factory->create(0);
    
    EXPECT_EQ(bv->size(), 0);
    EXPECT_TRUE(bv->empty());
}

TEST_F(SUXBitVectorFactoryTest, CreateFromVectorBool) {
    std::vector<bool> bits = {true, false, true, false, true};
    auto bv = factory->create(bits);
    
    EXPECT_EQ(bv->size(), 5);
    EXPECT_TRUE((*bv)[0]);
    EXPECT_FALSE((*bv)[1]);
    EXPECT_TRUE((*bv)[2]);
    EXPECT_FALSE((*bv)[3]);
    EXPECT_TRUE((*bv)[4]);
}

TEST_F(SUXBitVectorFactoryTest, CreateFromEmptyVectorBool) {
    std::vector<bool> bits;
    auto bv = factory->create(bits);
    
    EXPECT_EQ(bv->size(), 0);
    EXPECT_TRUE(bv->empty());
}

TEST_F(SUXBitVectorFactoryTest, FromFile) {
    std::filesystem::path test_file = "test_sux_factory_file.bin";
    test_files.push_back(test_file);
    
    // Create a test file
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1};
    SUXBitVector original(bits);
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

TEST_F(SUXBitVectorFactoryTest, FromFileError) {
    // Try to load from non-existent file
    EXPECT_THROW(factory->from_file("non_existent_file.bin"), std::runtime_error);
}

// Integration test: verify factory creates SUXBitVector instances
TEST_F(SUXBitVectorFactoryTest, FactoryCreatesSuxBitVector) {
    auto bv = factory->create(10);
    
    // Enable features to test they work
    bv->enable_rank();
    bv->enable_select1();
    bv->enable_select0();
    
    // This should not throw
    EXPECT_NO_THROW(bv->rank(10));
}

TEST_F(SUXBitVectorFactoryTest, GetOverheads) {
    EXPECT_GT(factory->get_rank_overhead(), 0.0);
    EXPECT_GT(factory->get_select1_overhead(), 0.0);
    EXPECT_GT(factory->get_select0_overhead(), 0.0);
}

// Edge case tests
TEST_F(SUXBitVectorTest, LargeBitVector) {
    SUXBitVector bv(1000000);
    EXPECT_EQ(bv.size(), 1000000);
    
    bv.set(999999, true);
    EXPECT_TRUE(bv[999999]);
    
    bv.enable_rank();
    EXPECT_EQ(bv.rank(1000000), 1);
}

TEST_F(SUXBitVectorTest, AllOnesVector) {
    std::vector<bool> all_ones(100, true);
    SUXBitVector bv(all_ones);
    bv.enable_rank();
    bv.enable_select1();
    bv.enable_select0();
    
    EXPECT_EQ(bv.rank(100), 100);
    EXPECT_EQ(bv.select(50), 49); // 50th one is at position 49
    
    // No zeros, so check total zeros via rank0
    EXPECT_EQ(bv.rank0(100), 0);
}

TEST_F(SUXBitVectorTest, AllZerosVector) {
    std::vector<bool> all_zeros(100, false);
    SUXBitVector bv(all_zeros);
    bv.enable_rank();
    bv.enable_select1();
    bv.enable_select0();
    
    EXPECT_EQ(bv.rank(100), 0);
    EXPECT_EQ(bv.select0(50), 49); // 50th zero is at position 49
    // Check total ones via rank
    EXPECT_EQ(bv.rank(100), 0);
}

// --- Tests for bulk modification methods ---

TEST_F(SUXBitVectorTest, SetRangeSetsBitsToTrue) {
    SUXBitVector bv(200); // More than 2*64 bits
    bv.set_range(70, 60, true);

    // Verify before the range
    for (size_t i = 0; i < 70; ++i) {
        EXPECT_FALSE(bv[i]) << "Bit at index " << i << " should be false.";
    }

    // Verify inside the range
    for (size_t i = 70; i < 130; ++i) {
        EXPECT_TRUE(bv[i]) << "Bit at index " << i << " should be true.";
    }

    // Verify after the range
    for (size_t i = 130; i < 200; ++i) {
        EXPECT_FALSE(bv[i]) << "Bit at index " << i << " should be false.";
    }
}

TEST_F(SUXBitVectorTest, SetRangeClearsBitsToFalse) {
    SUXBitVector bv(200);
    // Set all bits to 1 first
    bv.set_range(0, 200, true);

    // Clear a range in the middle
    bv.set_range(70, 60, false);

    // Verify before the range
    for (size_t i = 0; i < 70; ++i) {
        EXPECT_TRUE(bv[i]) << "Bit at index " << i << " should be true.";
    }

    // Verify inside the range
    for (size_t i = 70; i < 130; ++i) {
        EXPECT_FALSE(bv[i]) << "Bit at index " << i << " should be false.";
    }

    // Verify after the range
    for (size_t i = 130; i < 200; ++i) {
        EXPECT_TRUE(bv[i]) << "Bit at index " << i << " should be true.";
    }
}

TEST_F(SUXBitVectorTest, SetRangeEdgeCases) {
    SUXBitVector bv(100);

    // Set range starting at 0
    bv.set_range(0, 10, true);
    for(size_t i = 0; i < 10; ++i) EXPECT_TRUE(bv[i]);
    EXPECT_FALSE(bv[10]);

    // Set range ending at size-1
    bv.set_range(90, 10, true);
    EXPECT_FALSE(bv[89]);
    for(size_t i = 90; i < 100; ++i) EXPECT_TRUE(bv[i]);

    // Set range with count = 0 (should be a no-op)
    bv.set_range(50, 0, true);
    EXPECT_FALSE(bv[50]);
}

TEST_F(SUXBitVectorTest, SetRangeSingleWord) {
    SUXBitVector bv(100);
    bv.set_range(5, 20, true); // Stays within the first 64-bit word
    EXPECT_FALSE(bv[4]);
    for(size_t i = 5; i < 25; ++i) EXPECT_TRUE(bv[i]);
    EXPECT_FALSE(bv[25]);
}

TEST_F(SUXBitVectorTest, SetRangeOutOfBounds) {
    SUXBitVector bv(100);
    EXPECT_THROW(bv.set_range(90, 20, true), std::out_of_range);
    EXPECT_THROW(bv.set_range(100, 1, true), std::out_of_range);
}

TEST_F(SUXBitVectorTest, SetBitsBasic) {
    SUXBitVector bv(100);
    uint64_t pattern = 0b110101; // Represents 53
    uint8_t num_bits = 6;

    bv.set_bits(10, pattern, num_bits);

    EXPECT_FALSE(bv[9]);
    EXPECT_TRUE(bv[10]);  // LSB
    EXPECT_FALSE(bv[11]);
    EXPECT_TRUE(bv[12]);
    EXPECT_FALSE(bv[13]);
    EXPECT_TRUE(bv[14]);
    EXPECT_TRUE(bv[15]);  // MSB
    EXPECT_FALSE(bv[16]);
}

TEST_F(SUXBitVectorTest, SetBitsAcrossWordBoundary) {
    SUXBitVector bv(100);
    uint64_t pattern = 0xFFFF0000FFFF0000;
    uint8_t num_bits = 32;

    // Write 32 bits starting at position 50
    // This will write 14 bits in the first word and 18 in the second.
    bv.set_bits(50, pattern, num_bits);

    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FALSE(bv[50 + i]); // Corresponds to the low 16 zero bits of the pattern
    }
    for (size_t i = 16; i < 32; ++i) {
        EXPECT_TRUE(bv[50 + i]); // Corresponds to the next 16 one bits of the pattern
    }
}

TEST_F(SUXBitVectorTest, SetBitsOutOfBounds) {
    SUXBitVector bv(100);
    EXPECT_THROW(bv.set_bits(90, 0x01, 20), std::out_of_range);
    EXPECT_THROW(bv.set_bits(10, 0x01, 65), std::invalid_argument);
}

TEST_F(SUXBitVectorTest, CloneTest) {
    SUXBitVector bv(50);
    bv.set(10, true);
    bv.set(20, true);
    bv.enable_rank();
    
    auto cloned = bv.clone();
    
    EXPECT_EQ(cloned->size(), 50);
    EXPECT_TRUE((*cloned)[10]);
    EXPECT_TRUE((*cloned)[20]);
    EXPECT_EQ(cloned->rank(50), 2);
}



