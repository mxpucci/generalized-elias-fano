#include <gtest/gtest.h>
#include "gef/U_GEF.hpp"
#include "gef/B_GEF.hpp"
#include "gef/B_STAR_GEF.hpp"
#include "gef_test_utils.hpp"
#include <vector>
#include <type_traits>
#include <typeinfo>

#include "gef/gap_computation_utils.hpp"
#include "split_point_utils.hpp"

// Helper trait to extract the underlying value_type from a GEF implementation class.
// (Moved to gef_test_utils.hpp)

// Helper trait to check if a type is a specialization of B_GEF_NO_RLE.
template <typename T>
struct is_b_gef_no_rle : std::false_type {};

template <typename ValType>
struct is_b_gef_no_rle<gef::internal::B_STAR_GEF<ValType>> : std::true_type {};


// Test fixture for typed tests.
template <typename T>
class GEF_SplitPointStrategyComparison_TypedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // no setup needed
    }
};

namespace {
// Define the list of implementations and types to test with.
using GEF_SplitStrategyComparison_Implementations = ::testing::Types<
    gef::internal::U_GEF<int16_t>, gef::internal::B_GEF<int16_t>, gef::internal::B_STAR_GEF<int16_t>,
    gef::internal::U_GEF<uint16_t>, gef::internal::B_GEF<uint16_t>, gef::internal::B_STAR_GEF<uint16_t>,
    gef::internal::U_GEF<int32_t>, gef::internal::B_GEF<int32_t>, gef::internal::B_STAR_GEF<int32_t>,
    gef::internal::U_GEF<uint32_t>, gef::internal::B_GEF<uint32_t>, gef::internal::B_STAR_GEF<uint32_t>
>;
}

TYPED_TEST_CASE(GEF_SplitPointStrategyComparison_TypedTest, GEF_SplitStrategyComparison_Implementations);

TYPED_TEST(GEF_SplitPointStrategyComparison_TypedTest, BruteForceIsOptimal) {
    using GEF_Class = TypeParam;
    using value_type = typename get_value_type<GEF_Class>::type;

    struct TestCase {
        std::string name;
        std::vector<value_type> sequence;
    };

    std::vector<TestCase> test_cases;

    // Highly compressible sequence: small gaps, mostly increasing
    test_cases.push_back({
        "Highly Compressible",
        gef::test::generate_random_sequence<value_type>(2000, 0, 100, 0.8, 10)
    });

    // Slightly compressible sequence: medium gaps, some noise
    test_cases.push_back({
        "Slightly Compressible",
        gef::test::generate_random_sequence<value_type>(2000, 0, 1000, 0.3, 3)
    });

    // Poorly compressible sequence: large gaps, random data
    test_cases.push_back({
        "Poorly Compressible",
        gef::test::generate_random_sequence<value_type>(2000, 0, 30000, 0.0, 1)
    });

    // All elements are the same
    test_cases.push_back({
        "All Elements Identical",
        std::vector<value_type>(1000, 42)
    });

    // Alternating values
    std::vector<value_type> alternating_seq;
    alternating_seq.reserve(1000);
    for(int i = 0; i < 1000; ++i) {
        alternating_seq.push_back(i % 2 == 0 ? 10 : 1000);
    }
    test_cases.push_back({ "Alternating Values", alternating_seq });


    for (const auto& test : test_cases) {
        SCOPED_TRACE("Test Case: " + test.name);

        // Instantiate with Optimal strategy
        GEF_Class gef_brute_force(test.sequence, gef::OPTIMAL_SPLIT_POINT);
        size_t brute_force_size = gef_brute_force.theoretical_size_in_bytes();

        // Instantiate with Approximate strategy
        GEF_Class gef_approximate(test.sequence, gef::APPROXIMATE_SPLIT_POINT);
        size_t approximate_size = gef_approximate.theoretical_size_in_bytes();


        // The approximate strategy should always be larger or equal in size than brute force.
        EXPECT_LE(brute_force_size, approximate_size)
            << "[" << test.name << "] " << "Brute force size (" << brute_force_size
            << ") should be <= approximate size (" << approximate_size << ")";


        // Also verify correctness of the data
        for(size_t i = 0; i < test.sequence.size(); ++i) {
            ASSERT_EQ(gef_brute_force.at(i), test.sequence[i]);
            ASSERT_EQ(gef_approximate.at(i), test.sequence[i]);
        }
    }
}

namespace {

template<typename T>
void assert_optimal_not_exceeded(const std::vector<T>& sequence,
                                 const std::string& label) {
    ASSERT_FALSE(sequence.empty()) << "Sequence should not be empty for " << label;

    gef::internal::B_STAR_GEF<T> optimal(sequence, gef::OPTIMAL_SPLIT_POINT);
    const size_t optimal_size = optimal.theoretical_size_in_bytes();

    auto [min_it, max_it] = std::minmax_element(sequence.begin(), sequence.end());
    const size_t total_bits = gef::test::compute_total_bits_from_range(*min_it, *max_it);

    for (size_t b = 0; b <= 64; ++b) {
        const size_t candidate_size = gef::test::theoretical_size_for_split(sequence,
                                                                            *min_it,
                                                                            *max_it,
                                                                            total_bits,
                                                                            static_cast<uint8_t>(b));
        EXPECT_LE(optimal_size, candidate_size)
            << "[" << label << "] "
            << "type=" << typeid(T).name()
            << " optimal_size=" << optimal_size
            << " candidate_size=" << candidate_size
            << " candidate_b=" << b;
    }
}

template<typename T>
void run_full_sweep_assertions() {
    struct TestCase {
        std::string name;
        std::vector<T> sequence;
    };

    std::vector<TestCase> test_cases;
    test_cases.push_back({
        "Highly Compressible",
        gef::test::generate_random_sequence<T>(2000, 0, 100, 0.8, 10)
    });
    test_cases.push_back({
        "Slightly Compressible",
        gef::test::generate_random_sequence<T>(2000, 0, 1000, 0.3, 3)
    });
    test_cases.push_back({
        "Poorly Compressible",
        gef::test::generate_random_sequence<T>(2000, 0, 30000, 0.0, 1)
    });
    test_cases.push_back({
        "All Elements Identical",
        std::vector<T>(1000, static_cast<T>(42))
    });

    std::vector<T> alternating_seq;
    alternating_seq.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        alternating_seq.push_back(i % 2 == 0 ? static_cast<T>(10) : static_cast<T>(1000));
    }
    test_cases.push_back({"Alternating Values", alternating_seq});

    for (const auto& test : test_cases) {
        SCOPED_TRACE("Full sweep test: " + test.name);
        assert_optimal_not_exceeded(test.sequence, test.name);
    }
}

} // namespace

TEST(B_STAR_GEF_SplitPointStrategy, OptimalDominatesAllSplitPoints) {
    run_full_sweep_assertions<int32_t>();
    run_full_sweep_assertions<uint32_t>();
    run_full_sweep_assertions<int64_t>();
    run_full_sweep_assertions<uint64_t>();
}

template<typename T>
void expect_split_point_variation() {
    bool difference_observed = false;
    const size_t max_attempts = 200;

    for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
        auto sequence = gef::test::generate_random_sequence<T>(
            2000,
            0,
            100000,
            0.5,
            5
        );

        if (sequence.empty()) {
            continue;
        }

        gef::internal::B_STAR_GEF<T> approx(sequence, gef::APPROXIMATE_SPLIT_POINT);
        gef::internal::B_STAR_GEF<T> optimal(sequence, gef::OPTIMAL_SPLIT_POINT);

        if (approx.split_point() != optimal.split_point()) {
            difference_observed = true;
            break;
        }
    }

    EXPECT_TRUE(difference_observed)
        << "Approximate and optimal split points were identical for all "
        << max_attempts << " random sequences of type " << typeid(T).name()
        << ". This suggests the approximation may be degenerating to the exact strategy.";
}

/*
TEST(B_STAR_GEF_SplitPointStrategy, ApproximateOccasionallyDiffers) {
    expect_split_point_variation<int32_t>();
    expect_split_point_variation<uint32_t>();
    expect_split_point_variation<int64_t>();
    expect_split_point_variation<uint64_t>();
}
*/