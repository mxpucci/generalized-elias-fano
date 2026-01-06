#include <gtest/gtest.h>

#include "gef/B_STAR_GEF.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "gef/utils.hpp"

#include <cctype>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

namespace {

struct DatasetOptions {
    std::string dataset_path;
    std::string type = "int64";
    bool first_is_size = true;
    size_t max_elements = std::numeric_limits<size_t>::max();
};

DatasetOptions& global_options() {
    static DatasetOptions opts;
    return opts;
}

bool parse_bool(std::string_view value) {
    if (value == "1" || value == "true" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "off") {
        return false;
    }
    throw std::invalid_argument("Invalid boolean value: " + std::string(value));
}

bool parse_dataset_cli(int* argc, char** argv) {
    auto& opts = global_options();
    int write_idx = 1;
    for (int i = 1; i < *argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg.rfind("--dataset=", 0) == 0) {
            opts.dataset_path = std::string(arg.substr(10));
            continue;
        }
        if (arg.rfind("--type=", 0) == 0) {
            auto type_value = arg.substr(7);
            std::string lowered(type_value);
            for (auto& c : lowered) c = static_cast<char>(std::tolower(c));
            opts.type = lowered;
            continue;
        }
        if (arg.rfind("--first-is-size=", 0) == 0) {
            opts.first_is_size = parse_bool(arg.substr(16));
            continue;
        }
        if (arg.rfind("--max-elements=", 0) == 0) {
            opts.max_elements = static_cast<size_t>(std::stoull(std::string(arg.substr(15))));
            continue;
        }

        argv[write_idx++] = argv[i];
    }
    *argc = write_idx;
    return true;
}

template <typename T>
void run_dataset_comparison(const DatasetOptions& opts) {
    auto factory = std::make_shared<SDSLBitVectorFactory>();

    auto data = read_data_binary<T, T>(opts.dataset_path, opts.first_is_size, opts.max_elements);
    ASSERT_FALSE(data.empty()) << "Dataset " << opts.dataset_path << " is empty";

    gef::B_STAR_GEF<T> approx(factory, data, gef::APPROXIMATE_SPLIT_POINT);
    gef::B_STAR_GEF<T> optimal(factory, data, gef::OPTIMAL_SPLIT_POINT);

    const size_t approx_size = approx.theoretical_size_in_bytes();
    const size_t optimal_size = optimal.theoretical_size_in_bytes();

    std::cout << "Dataset: " << opts.dataset_path << "\n"
              << "Elements: " << data.size() << "\n"
              << "First value stores size: " << std::boolalpha << opts.first_is_size << "\n"
              << "Approx split point:  " << static_cast<int>(approx.split_point()) << "\n"
              << "Optimal split point: " << static_cast<int>(optimal.split_point()) << "\n"
              << "Approx bytes:  " << approx_size << "\n"
              << "Optimal bytes: " << optimal_size << "\n"
              << "Difference:    " << static_cast<long long>(approx_size) - static_cast<long long>(optimal_size)
              << "\n"
              << std::endl;

    EXPECT_LE(optimal_size, approx_size)
        << "Optimal strategy should not produce a larger structure than approximate.";
}

void run_for_type(const DatasetOptions& opts) {
    const std::string& type = opts.type;
    if (type == "int32") {
        run_dataset_comparison<int32_t>(opts);
    } else if (type == "uint32") {
        run_dataset_comparison<uint32_t>(opts);
    } else if (type == "int64") {
        run_dataset_comparison<int64_t>(opts);
    } else if (type == "uint64") {
        run_dataset_comparison<uint64_t>(opts);
    } else if (type == "int16") {
        run_dataset_comparison<int16_t>(opts);
    } else if (type == "uint16") {
        run_dataset_comparison<uint16_t>(opts);
    } else if (type == "int8") {
        run_dataset_comparison<int8_t>(opts);
    } else if (type == "uint8") {
        run_dataset_comparison<uint8_t>(opts);
    } else {
        FAIL() << "Unsupported --type value: " << type;
    }
}

} // namespace

TEST(SplitPointDatasetTest, CompareApproximateAndOptimal) {
    const auto& opts = global_options();
    if (opts.dataset_path.empty()) {
        std::cout << "[ SKIPPED ] Provide --dataset=/path/to/data.bin to run this test." << std::endl;
        return;
    }
    run_for_type(opts);
}

int main(int argc, char** argv) {
    parse_dataset_cli(&argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}