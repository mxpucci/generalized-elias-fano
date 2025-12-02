#include <gtest/gtest.h>
#include "../include/datastructures/PastaBitVector.hpp"
#include "../include/datastructures/PastaBitVectorFactory.hpp"
#include <filesystem>
#include <fstream>
#include <vector>

class PastaBitVectorTest : public ::testing::Test {
protected:
    std::vector<std::filesystem::path> test_files;

    void TearDown() override {
        for (const auto& path : test_files) {
            if (std::filesystem::exists(path)) {
                std::filesystem::remove(path);
            }
        }
    }
};

// --- Constructors ---
TEST_F(PastaBitVectorTest, ConstructorFromSize) {
    PastaBitVector bv(100);
    EXPECT_EQ(bv.size(), 100);
    EXPECT_FALSE(bv.empty());

    for (size_t i = 0; i < bv.size(); ++i) {
        EXPECT_FALSE(bv[i]);
    }
}

TEST_F(PastaBitVectorTest, ConstructorFromSizeZero) {
    PastaBitVector bv(0);
    EXPECT_EQ(bv.size(), 0);
    EXPECT_TRUE(bv.empty());
}

TEST_F(PastaBitVectorTest, ConstructorFromVectorBool) {
    std::vector<bool> bits = {true, false, true, false, true};
    PastaBitVector bv(bits);

    EXPECT_EQ(bv.size(), bits.size());
    EXPECT_TRUE(bv[0]);
    EXPECT_FALSE(bv[1]);
    EXPECT_TRUE(bv[2]);
    EXPECT_FALSE(bv[3]);
    EXPECT_TRUE(bv[4]);
}

TEST_F(PastaBitVectorTest, ConstructorFromEmptyVectorBool) {
    std::vector<bool> bits;
    PastaBitVector bv(bits);

    EXPECT_EQ(bv.size(), 0);
    EXPECT_TRUE(bv.empty());
}

// --- Bit access ---
TEST_F(PastaBitVectorTest, BitAccessAndModification) {
    PastaBitVector bv(10);

    bv.set(0, true);
    bv.set(5, true);
    bv[9] = true;

    const PastaBitVector& const_bv = bv;
    EXPECT_TRUE(const_bv[0]);
    EXPECT_FALSE(const_bv[1]);
    EXPECT_TRUE(const_bv[5]);
    EXPECT_TRUE(const_bv[9]);
}

// --- Rule of 5 ---
TEST_F(PastaBitVectorTest, CopyConstructorWithSupport) {
    PastaBitVector original(20);
    original[5] = true;
    original[10] = true;
    original.enable_rank();
    original.enable_select1();
    original.enable_select0();

    PastaBitVector copy(original);

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_TRUE(copy[5]);
    EXPECT_TRUE(copy[10]);
    EXPECT_EQ(copy.rank(20), 2);
    EXPECT_EQ(copy.select(1), 5);
    EXPECT_EQ(copy.select0(1), 0);
}

TEST_F(PastaBitVectorTest, CopyConstructorWithoutSupport) {
    PastaBitVector original(10);
    original[3] = true;

    PastaBitVector copy(original);

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_TRUE(copy[3]);
    // Note: Calling rank/select without enabling support is undefined behavior
    // The caller is responsible for enabling supports before use
}

TEST_F(PastaBitVectorTest, CopyAssignment) {
    PastaBitVector original(15);
    original[7] = true;
    original.enable_rank();

    PastaBitVector copy(5);
    copy = original;

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_TRUE(copy[7]);
    EXPECT_EQ(copy.rank(15), 1);
}

TEST_F(PastaBitVectorTest, CopyAssignmentSelf) {
    PastaBitVector bv(10);
    bv[3] = true;
    bv.enable_rank();

    bv = bv;

    EXPECT_EQ(bv.size(), 10);
    EXPECT_TRUE(bv[3]);
    EXPECT_EQ(bv.rank(10), 1);
}

TEST_F(PastaBitVectorTest, MoveConstructor) {
    PastaBitVector original(25);
    original[12] = true;
    original.enable_rank();
    original.enable_select0();

    PastaBitVector moved(std::move(original));

    EXPECT_EQ(moved.size(), 25);
    EXPECT_TRUE(moved[12]);
    EXPECT_EQ(moved.rank(25), 1);
    EXPECT_EQ(moved.select0(1), 0);
}

TEST_F(PastaBitVectorTest, MoveAssignment) {
    PastaBitVector original(30);
    original[15] = true;
    original.enable_rank();

    PastaBitVector moved(5);
    moved = std::move(original);

    EXPECT_EQ(moved.size(), 30);
    EXPECT_TRUE(moved[15]);
    EXPECT_EQ(moved.rank(30), 1);
}

TEST_F(PastaBitVectorTest, MoveAssignmentSelf) {
    PastaBitVector bv(10);
    bv[3] = true;
    bv.enable_rank();

    bv = std::move(bv);

    EXPECT_EQ(bv.size(), 10);
    EXPECT_TRUE(bv[3]);
    EXPECT_EQ(bv.rank(10), 1);
}

// --- Rank & Select ---
TEST_F(PastaBitVectorTest, RankOperations) {
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    PastaBitVector bv(bits);
    bv.enable_rank();

    EXPECT_EQ(bv.rank(0), 0);
    EXPECT_EQ(bv.rank(1), 1);
    EXPECT_EQ(bv.rank(3), 2);
    EXPECT_EQ(bv.rank(6), 4);
    EXPECT_EQ(bv.rank(10), 6);

}

// RankWithoutSupport test removed - calling rank without enabling support is undefined behavior
// The caller is responsible for enabling supports before use

TEST_F(PastaBitVectorTest, RankInheritedMethods) {
    std::vector<bool> bits = {1, 0, 1, 1, 0};
    PastaBitVector bv(bits);
    bv.enable_rank();

    EXPECT_EQ(bv.rank0(5), 2);
    EXPECT_EQ(bv.rank0(1, 4), 1);
    EXPECT_EQ(bv.rank(4), 3);
}

TEST_F(PastaBitVectorTest, SelectOperations) {
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    PastaBitVector bv(bits);
    bv.enable_select1();

    EXPECT_EQ(bv.select(1), 0);
    EXPECT_EQ(bv.select(2), 2);
    EXPECT_EQ(bv.select(3), 3);
    EXPECT_EQ(bv.select(4), 5);
    EXPECT_EQ(bv.select(5), 8);
    EXPECT_EQ(bv.select(6), 9);
}

TEST_F(PastaBitVectorTest, Select0Operations) {
    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    PastaBitVector bv(bits);
    bv.enable_select0();

    EXPECT_EQ(bv.select0(1), 1);
    EXPECT_EQ(bv.select0(2), 4);
    EXPECT_EQ(bv.select0(3), 6);
    EXPECT_EQ(bv.select0(4), 7);
}

// SelectWithoutSupport test removed - calling select without enabling support is undefined behavior
// The caller is responsible for enabling supports before use

TEST_F(PastaBitVectorTest, EnableMethods) {
    PastaBitVector bv(10);
    bv[5] = true;

    // Enable supports before use (calling without support is undefined behavior)
    bv.enable_rank();
    EXPECT_EQ(bv.rank(10), 1);
    EXPECT_EQ(bv.select(1), 5);
    EXPECT_EQ(bv.select0(1), 0);

    // Idempotent calls
    bv.enable_select1();
    bv.enable_select0();
    EXPECT_EQ(bv.rank(10), 1);
    EXPECT_EQ(bv.select(1), 5);
    EXPECT_EQ(bv.select0(1), 0);
}

// --- Memory & serialization ---
TEST_F(PastaBitVectorTest, MemoryOperations) {
    PastaBitVector bv(512);
    const size_t base_size = bv.size_in_bytes();
    EXPECT_GT(base_size, 0u);
    EXPECT_EQ(bv.support_size_in_bytes(), 0u);

    bv.enable_rank();
    const size_t with_support = bv.size_in_bytes();
    EXPECT_GT(with_support, base_size);
    EXPECT_EQ(bv.support_size_in_bytes(), with_support - base_size);

    bv.enable_select1();
    bv.enable_select0();
    EXPECT_EQ(bv.size_in_bytes(), with_support);

    size_t mb = bv.size_in_megabytes();
    EXPECT_EQ(mb, (with_support + (1024 * 1024 - 1)) / (1024 * 1024));
}

TEST_F(PastaBitVectorTest, SerializationAndLoading) {
    std::filesystem::path test_file = "test_pasta_bitvector.bin";
    test_files.push_back(test_file);

    std::vector<bool> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    PastaBitVector original(bits);
    original.enable_rank();
    original.serialize(test_file);

    PastaBitVector loaded = PastaBitVector::load(test_file);
    EXPECT_EQ(loaded.size(), original.size());
    for (size_t i = 0; i < loaded.size(); ++i) {
        EXPECT_EQ(static_cast<bool>(loaded[i]), static_cast<bool>(original[i]));
    }
    EXPECT_EQ(loaded.rank(10), 6);
}

TEST_F(PastaBitVectorTest, SerializationError) {
    PastaBitVector bv(10);
    EXPECT_THROW(bv.serialize("/invalid/path/file.bin"), std::runtime_error);
}

TEST_F(PastaBitVectorTest, LoadError) {
    EXPECT_THROW(PastaBitVector::load("missing_file.bin"), std::runtime_error);
}

// --- Edge cases ---
TEST_F(PastaBitVectorTest, LargeBitVector) {
    PastaBitVector bv(100000);
    EXPECT_EQ(bv.size(), 100000);
    bv[99999] = true;
    EXPECT_TRUE(bv[99999]);
    bv.enable_rank();
    EXPECT_EQ(bv.rank(100000), 1);
}

TEST_F(PastaBitVectorTest, AllOnesVector) {
    std::vector<bool> ones(100, true);
    PastaBitVector bv(ones);
    bv.enable_rank();
    EXPECT_EQ(bv.rank(100), 100);
    bv.enable_select1();
    EXPECT_EQ(bv.select(50), 49);
    EXPECT_EQ(bv.rank0(100), 0);
}

TEST_F(PastaBitVectorTest, AllZerosVector) {
    std::vector<bool> zeros(100, false);
    PastaBitVector bv(zeros);
    bv.enable_rank();
    bv.enable_select0();
    EXPECT_EQ(bv.rank(100), 0);
    EXPECT_EQ(bv.select0(50), 49);
}

// --- Range & bulk operations ---
TEST_F(PastaBitVectorTest, SetRangeSetsBitsToTrue) {
    PastaBitVector bv(200);
    bv.set_range(70, 60, true);

    for (size_t i = 0; i < 70; ++i) {
        EXPECT_FALSE(bv[i]);
    }
    for (size_t i = 70; i < 130; ++i) {
        EXPECT_TRUE(bv[i]);
    }
    for (size_t i = 130; i < 200; ++i) {
        EXPECT_FALSE(bv[i]);
    }
}

TEST_F(PastaBitVectorTest, SetRangeClearsBitsToFalse) {
    PastaBitVector bv(200);
    bv.set_range(0, 200, true);
    bv.set_range(70, 60, false);

    for (size_t i = 0; i < 70; ++i) {
        EXPECT_TRUE(bv[i]);
    }
    for (size_t i = 70; i < 130; ++i) {
        EXPECT_FALSE(bv[i]);
    }
    for (size_t i = 130; i < 200; ++i) {
        EXPECT_TRUE(bv[i]);
    }
}

TEST_F(PastaBitVectorTest, SetRangeEdgeCases) {
    PastaBitVector bv(100);
    bv.set_range(0, 10, true);
    for (size_t i = 0; i < 10; ++i) EXPECT_TRUE(bv[i]);
    EXPECT_FALSE(bv[10]);

    bv.set_range(90, 10, true);
    EXPECT_FALSE(bv[89]);
    for (size_t i = 90; i < 100; ++i) EXPECT_TRUE(bv[i]);

    bv.set_range(50, 0, true);
    EXPECT_FALSE(bv[50]);
}

TEST_F(PastaBitVectorTest, SetRangeSingleWord) {
    PastaBitVector bv(128);
    bv.set_range(5, 20, true);
    EXPECT_FALSE(bv[4]);
    for (size_t i = 5; i < 25; ++i) EXPECT_TRUE(bv[i]);
    EXPECT_FALSE(bv[25]);
}

TEST_F(PastaBitVectorTest, SetBitsBasic) {
    PastaBitVector bv(100);
    uint64_t pattern = 0b110101;
    uint8_t num_bits = 6;

    bv.set_bits(10, pattern, num_bits);

    EXPECT_FALSE(bv[9]);
    EXPECT_TRUE(bv[10]);
    EXPECT_FALSE(bv[11]);
    EXPECT_TRUE(bv[12]);
    EXPECT_FALSE(bv[13]);
    EXPECT_TRUE(bv[14]);
    EXPECT_TRUE(bv[15]);
    EXPECT_FALSE(bv[16]);
}

TEST_F(PastaBitVectorTest, SetBitsAcrossWordBoundary) {
    PastaBitVector bv(128);
    uint64_t pattern = 0xFFFF0000FFFF0000ULL;
    uint8_t num_bits = 32;

    bv.set_bits(60, pattern, num_bits);

    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FALSE(bv[60 + i]);
    }
    for (size_t i = 16; i < 32; ++i) {
        EXPECT_TRUE(bv[60 + i]);
    }
}

TEST_F(PastaBitVectorTest, CloneCreatesIndependentCopy) {
    PastaBitVector bv(50);
    bv[10] = true;
    bv[20] = true;
    bv.enable_rank();

    auto clone = bv.clone();

    EXPECT_EQ(clone->size(), 50);
    EXPECT_TRUE((*clone)[10]);
    EXPECT_TRUE((*clone)[20]);
    EXPECT_EQ(clone->rank(50), 2);

    clone->set(10, false);
    EXPECT_FALSE((*clone)[10]);
    EXPECT_TRUE(bv[10]);
}

// --- Factory tests ---
class PastaBitVectorFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        factory = std::make_unique<PastaBitVectorFactory>();
    }

    void TearDown() override {
        for (const auto& path : test_files) {
            if (std::filesystem::exists(path)) {
                std::filesystem::remove(path);
            }
        }
    }

    std::unique_ptr<PastaBitVectorFactory> factory;
    std::vector<std::filesystem::path> test_files;
};

TEST_F(PastaBitVectorFactoryTest, CreateFromSize) {
    auto bv = factory->create(100);
    EXPECT_EQ(bv->size(), 100);
    EXPECT_FALSE(bv->empty());
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FALSE((*bv)[i]);
    }
}

TEST_F(PastaBitVectorFactoryTest, CreateFromSizeZero) {
    auto bv = factory->create(0);
    EXPECT_EQ(bv->size(), 0);
    EXPECT_TRUE(bv->empty());
}

TEST_F(PastaBitVectorFactoryTest, CreateFromVectorBool) {
    std::vector<bool> bits = {true, false, true, false, true};
    auto bv = factory->create(bits);
    EXPECT_EQ(bv->size(), bits.size());
    EXPECT_TRUE((*bv)[0]);
    EXPECT_FALSE((*bv)[1]);
    EXPECT_TRUE((*bv)[2]);
    EXPECT_FALSE((*bv)[3]);
    EXPECT_TRUE((*bv)[4]);
}

TEST_F(PastaBitVectorFactoryTest, CreateFromEmptyVectorBool) {
    std::vector<bool> bits;
    auto bv = factory->create(bits);
    EXPECT_EQ(bv->size(), 0);
    EXPECT_TRUE(bv->empty());
}

TEST_F(PastaBitVectorFactoryTest, FromFile) {
    std::filesystem::path test_file = "test_pasta_factory.bin";
    test_files.push_back(test_file);

    std::vector<bool> bits = {1, 0, 1, 1, 0, 1};
    PastaBitVector original(bits);
    original.enable_rank();
    original.serialize(test_file);

    auto loaded = factory->from_file(test_file);
    EXPECT_EQ(loaded->size(), bits.size());
    EXPECT_TRUE((*loaded)[0]);
    EXPECT_FALSE((*loaded)[1]);
    EXPECT_TRUE((*loaded)[2]);
    EXPECT_TRUE((*loaded)[3]);
    EXPECT_FALSE((*loaded)[4]);
    EXPECT_TRUE((*loaded)[5]);
}

TEST_F(PastaBitVectorFactoryTest, FromFileError) {
    EXPECT_THROW(factory->from_file("non_existent_file.bin"), std::runtime_error);
}

TEST_F(PastaBitVectorFactoryTest, FactoryCreatesReadyBitVector) {
    // Create with at least one 1-bit so select(1) is valid
    // (support is built at creation time, so we can't modify bits after)
    std::vector<bool> bits = {false, false, false, false, false, true, false, false, false, false};
    auto bv = factory->create(bits);
    EXPECT_NO_THROW(bv->rank(10));
    EXPECT_NO_THROW(bv->select(1));  // Finds the 1-bit at position 5
    EXPECT_NO_THROW(bv->select0(1)); // Finds the 0-bit at position 0
}

TEST_F(PastaBitVectorFactoryTest, GetOverheads) {
    EXPECT_GT(factory->get_rank_overhead(), 0.0);
    EXPECT_GT(factory->get_select1_overhead(), 0.0);
    EXPECT_GT(factory->get_select0_overhead(), 0.0);
}


