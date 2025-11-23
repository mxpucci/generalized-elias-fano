#include <gtest/gtest.h>
#include "gef/RLE_GEF.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "gef_test_utils.hpp"
#include <vector>
#include <memory>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include "gef/U_GEF.hpp"
#include "gef/B_GEF.hpp"
#include "gef/B_GEF_STAR.hpp"

// Helper trait to extract the underlying value_type from a GEF implementation class.
// e.g., get_value_type<gef::RLE_GEF<int32_t>>::type will be int32_t.
template<typename T>
struct get_value_type;

template<template<typename> class C, typename T>
struct get_value_type<C<T>> {
    using type = T;
};


// Test fixture for typed tests. This allows us to run the same tests
// for different GEF implementations and integral types.
template <typename T>
class GEF_Implementation_TypedTest : public ::testing::Test {
protected:
    // A shared factory for creating bit vectors in tests.
    std::shared_ptr<IBitVectorFactory> factory;

    void SetUp() override {
        // Initialize the factory before each test.
        factory = std::make_shared<SDSLBitVectorFactory>();
    }
};

// Define the list of all implementations and types we want to test with.
using Implementations = ::testing::Types<
    gef::RLE_GEF<int8_t>, gef::U_GEF<int8_t>, gef::B_GEF<int8_t>, gef::B_GEF_STAR<int8_t>,
    gef::RLE_GEF<uint8_t>, gef::U_GEF<uint8_t>, gef::B_GEF<uint8_t>, gef::B_GEF_STAR<uint8_t>,
    gef::RLE_GEF<int16_t>, gef::U_GEF<int16_t>, gef::B_GEF<int16_t>, gef::B_GEF_STAR<int16_t>,
    gef::RLE_GEF<uint16_t>, gef::U_GEF<uint16_t>, gef::B_GEF<uint16_t>, gef::B_GEF_STAR<uint16_t>,
    gef::RLE_GEF<int32_t>, gef::U_GEF<int32_t>, gef::B_GEF<int32_t>, gef::B_GEF_STAR<int32_t>,
    gef::RLE_GEF<uint32_t>, gef::U_GEF<uint32_t>, gef::B_GEF<uint32_t>, gef::B_GEF_STAR<uint32_t>,
    gef::RLE_GEF<int64_t>, gef::U_GEF<int64_t>, gef::B_GEF<int64_t>, gef::B_GEF_STAR<int64_t>,
    gef::RLE_GEF<uint64_t>, gef::U_GEF<uint64_t>, gef::B_GEF<uint64_t>, gef::B_GEF_STAR<uint64_t>
>;

TYPED_TEST_CASE(GEF_Implementation_TypedTest, Implementations);

// --- Constructor and Rule of Five Tests ---

TYPED_TEST(GEF_Implementation_TypedTest, DefaultConstructor) {
    using GEF_Class = TypeParam;
    GEF_Class gef;
    EXPECT_EQ(gef.size(), 0);
    EXPECT_TRUE(gef.empty());
}

TYPED_TEST(GEF_Implementation_TypedTest, CopyConstructor) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    for (int k = 0; k < 5; ++k) {
        const auto min_val = std::is_signed_v<value_type> ? static_cast<value_type>(-100) : static_cast<value_type>(0);
        const auto max_val = static_cast<value_type>(100);
        const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(100, min_val, max_val);

        GEF_Class original(this->factory, sequence);
        GEF_Class copy(original);

        ASSERT_EQ(original.size(), copy.size());
        for(size_t i = 0; i < sequence.size(); ++i) {
            ASSERT_EQ(copy.at(i), sequence[i]) << "Mismatch in iteration " << k << " at index " << i;
        }
    }
}

TYPED_TEST(GEF_Implementation_TypedTest, CopyConstructorEmpty) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    const std::vector<value_type> empty_sequence = {};
    GEF_Class original_empty(this->factory, empty_sequence);
    GEF_Class copy_empty(original_empty);

    ASSERT_EQ(copy_empty.size(), 0);
    ASSERT_TRUE(copy_empty.empty());
}

TYPED_TEST(GEF_Implementation_TypedTest, CopyAssignment) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    for (int k = 0; k < 5; ++k) {
        const auto min1 = std::is_signed_v<value_type> ? static_cast<value_type>(-100) : static_cast<value_type>(0);
        const auto max1 = static_cast<value_type>(100);
        const std::vector<value_type> seq1 = gef::test::generate_random_sequence<value_type>(100, min1, max1);

        const auto min2 = static_cast<value_type>(200);
        const auto max2 = static_cast<value_type>(300);
        const std::vector<value_type> seq2 = gef::test::generate_random_sequence<value_type>(50, min2, max2);

        GEF_Class gef1(this->factory, seq1);
        GEF_Class gef2(this->factory, seq2);

        gef1 = gef2;

        ASSERT_EQ(gef1.size(), seq2.size());
         for(size_t i = 0; i < seq2.size(); ++i) {
            ASSERT_EQ(gef1.at(i), seq2[i]) << "Mismatch in iteration " << k << " at index " << i;
        }
    }
}

TYPED_TEST(GEF_Implementation_TypedTest, MoveConstructor) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    for (int k = 0; k < 5; ++k) {
        const auto min_val = std::is_signed_v<value_type> ? static_cast<value_type>(-100) : static_cast<value_type>(0);
        const auto max_val = static_cast<value_type>(100);
        const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(100, min_val, max_val);

        GEF_Class original(this->factory, sequence);
        GEF_Class moved_to(std::move(original));

        ASSERT_EQ(moved_to.size(), sequence.size());
        for(size_t i = 0; i < sequence.size(); ++i) {
            ASSERT_EQ(moved_to.at(i), sequence[i]) << "Mismatch in iteration " << k << " at index " << i;
        }
    }
}

TYPED_TEST(GEF_Implementation_TypedTest, MoveAssignment) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    for (int k = 0; k < 5; ++k) {
        const auto min1 = std::is_signed_v<value_type> ? static_cast<value_type>(-100) : static_cast<value_type>(0);
        const auto max1 = static_cast<value_type>(100);
        const std::vector<value_type> seq1 = gef::test::generate_random_sequence<value_type>(100, min1, max1);

        const auto min2 = static_cast<value_type>(200);
        const auto max2 = static_cast<value_type>(300);
        const std::vector<value_type> seq2 = gef::test::generate_random_sequence<value_type>(50, min2, max2);

        GEF_Class gef1(this->factory, seq1);
        GEF_Class gef2(this->factory, seq2);

        gef1 = std::move(gef2);

        ASSERT_EQ(gef1.size(), seq2.size());
        for(size_t i = 0; i < seq2.size(); ++i) {
            ASSERT_EQ(gef1.at(i), seq2[i]) << "Mismatch in iteration " << k << " at index " << i;
        }
    }
}

// --- Core Functionality and Edge Case Tests ---

TYPED_TEST(GEF_Implementation_TypedTest, EncodeEmpty) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;
    const std::vector<value_type> sequence = {};
    auto gef_impl = std::make_unique<GEF_Class>(this->factory, sequence);
    ASSERT_EQ(gef_impl->size(), 0);
}

TYPED_TEST(GEF_Implementation_TypedTest, AllElementsIdentical) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;
    const std::vector<value_type> sequence(100, 42);
    auto gef_impl = std::make_unique<GEF_Class>(this->factory, sequence);
    for(size_t i = 0; i < sequence.size(); ++i) {
        ASSERT_EQ(gef_impl->at(i), sequence[i]);
    }
}

TYPED_TEST(GEF_Implementation_TypedTest, AlternatingValues) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;
    std::vector<value_type> sequence;
    sequence.reserve(100);
    for(int i=0; i<100; ++i) {
        sequence.push_back(i % 2 == 0 ? 10 : 20);
    }
    auto gef_impl = std::make_unique<GEF_Class>(this->factory, sequence);
    for(size_t i = 0; i < sequence.size(); ++i) {
        ASSERT_EQ(gef_impl->at(i), sequence[i]);
    }
}

TYPED_TEST(GEF_Implementation_TypedTest, HighlyCompressibleSequence) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;
    for (int k = 0; k < 5; ++k) {
        std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(
            10000,
            10,
            127,
            0.5, // 30% chance of duplicates
            1000    // Allow up to 3 consecutive duplicates
        );

        auto gef_impl = std::make_unique<GEF_Class>(this->factory, sequence);
        for(size_t i = 0; i < sequence.size(); ++i) {
            ASSERT_EQ(gef_impl->at(i), sequence[i]) << "Mismatch in iteration " << k << " at index " << i;
        }
    }
}

TYPED_TEST(GEF_Implementation_TypedTest, GeneralPurposeSequence) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    for (int k = 0; k < 5; ++k) {
        value_type min_val = std::is_signed_v<value_type> ? static_cast<value_type>(-100) : static_cast<value_type>(0);
        if constexpr (sizeof(value_type) > 1) {
             min_val = std::is_signed_v<value_type> ? static_cast<value_type>(-500) : static_cast<value_type>(0);
        }
        value_type max_val = static_cast<value_type>(100);
        if constexpr (sizeof(value_type) > 1) {
             max_val = static_cast<value_type>(500);
        }
        const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(
            1000,
            min_val,
            max_val,
            0.3, // 30% chance of duplicates
            3    // Allow up to 3 consecutive duplicates
        );

        GEF_Class gef_impl(this->factory, sequence);

        for(size_t i = 0; i < sequence.size(); ++i) {
            ASSERT_EQ(gef_impl.at(i), sequence[i]) << "Mismatch in iteration " << k << " at index " << i;
        }
        ASSERT_GT(gef_impl.size_in_bytes(), 0);
        ASSERT_THROW(gef_impl.at(sequence.size()), std::out_of_range);
    }
}

TYPED_TEST(GEF_Implementation_TypedTest, EncodeLongSequence) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;
    for (int k = 0; k < 5; ++k) {
        value_type max_val = 100;
        if constexpr (sizeof(value_type) > 1) max_val = 1000;
        const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(10000, 0, max_val);
        auto gef_impl = std::make_unique<GEF_Class>(this->factory, sequence);
        for(size_t i = 0; i < sequence.size(); ++i) {
            ASSERT_EQ(gef_impl->at(i), sequence[i]) << "Mismatch in iteration " << k << " at index " << i;
        }
    }
}

TYPED_TEST(GEF_Implementation_TypedTest, DirectAccessOperator) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;
    for (int k = 0; k < 5; ++k) {
        const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(100, 0, 100);
        GEF_Class gef_impl(this->factory, sequence);
        for(size_t i = 0; i < sequence.size(); ++i) {
            ASSERT_EQ(gef_impl[i], sequence[i]) << "Mismatch in iteration " << k << " at index " << i;
        }
    }
}

TYPED_TEST(GEF_Implementation_TypedTest, SizeInMegabytes) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;
    for (int k = 0; k < 5; ++k) {
        const std::vector<value_type> sequence = gef::test::generate_random_sequence<value_type>(100, 0, 100);
        GEF_Class gef_impl(this->factory, sequence);
        double size_mb = gef_impl.size_in_megabytes();
        double size_b = static_cast<double>(gef_impl.size_in_bytes());
        EXPECT_NEAR(size_mb, size_b / (1024.0 * 1024.0), 1e-9);
    }
}

// --- Serialization Tests ---

TYPED_TEST(GEF_Implementation_TypedTest, SerializationDeserialization) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    for (int k = 0; k < 5; ++k) {
        value_type min_val = std::is_signed_v<value_type> ? static_cast<value_type>(-100) : static_cast<value_type>(0);
        value_type max_val = static_cast<value_type>(100);
        if constexpr (sizeof(value_type) > 1) {
             min_val = std::is_signed_v<value_type> ? static_cast<value_type>(-500) : static_cast<value_type>(0);
             max_val = static_cast<value_type>(500);
        }

        const std::vector<value_type> original_sequence = gef::test::generate_random_sequence<value_type>(
            1500, min_val, max_val, 0.5, 4
        );

        GEF_Class original_gef(this->factory, original_sequence);
        const std::filesystem::path temp_path = "temp_gef_impl_test.bin";

        std::ofstream ofs(temp_path, std::ios::binary);
        ASSERT_TRUE(ofs.is_open());
        original_gef.serialize(ofs);
        ofs.close();

        ASSERT_TRUE(std::filesystem::exists(temp_path));
        ASSERT_GT(std::filesystem::file_size(temp_path), 0);

        GEF_Class loaded_gef;
        std::ifstream ifs(temp_path, std::ios::binary);
        ASSERT_TRUE(ifs.is_open());
        loaded_gef.load(ifs, this->factory);
        ifs.close();

        ASSERT_EQ(original_gef.size(), loaded_gef.size());

        for (size_t i = 0; i < original_sequence.size(); ++i) {
            ASSERT_EQ(original_gef.at(i), loaded_gef.at(i)) << "Mismatch at index " << i << " in iteration " << k;
            ASSERT_EQ(original_sequence[i], loaded_gef.at(i)) << "Mismatch at index " << i << " in iteration " << k;
        }

        std::filesystem::remove(temp_path);
    }
}