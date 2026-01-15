#pragma once

#include <vector>
#include <string>
#include <benchmark/benchmark.h>
#include "gef/utils.hpp"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>

// Global input files list
inline std::vector<std::string> g_input_files;

// Helper to register input files from command line
inline void RegisterInputFiles(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        // Skip flags
        if (!arg.empty() && arg.rfind("-", 0) == 0) continue;
        if (std::filesystem::exists(arg)) {
            g_input_files.push_back(arg);
        }
    }
}

struct LoadedDataset {
    std::vector<int64_t> data;
    uint64_t decimals;
};

inline LoadedDataset load_custom_dataset(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    uint64_t n_val = 0;
    in.read(reinterpret_cast<char*>(&n_val), 8);
    size_t n = static_cast<size_t>(n_val);
    
    // Check if file size matches the new format (N + Decimals + Data)
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(8, std::ios::beg); // Skip N
    
    // Standard Format: 16 bytes Header (N, Decimals) + Data
    // Header (16 bytes): uint64_t N, uint64_t Decimals
    size_t expected_size_standard = 16 + n * 8;
    // Simple Format: 8 bytes Header (N) + Data
    // Header (8 bytes): uint64_t N
    size_t expected_size_simple = 8 + n * 8;
    
    uint64_t decimals = 0;
    std::vector<int64_t> data(n);
    
    if (file_size == expected_size_standard) {
        uint64_t dec_val = 0;
        in.read(reinterpret_cast<char*>(&dec_val), 8);
        decimals = dec_val;
    } else if (file_size == expected_size_simple) {
        // Simple format, no decimals parameter (defaults to 0)
        decimals = 0;
    } else {
        // Warning or error? Let's try to read data anyway if we can
        // Assuming simple format structure for safety if unknown
        std::cerr << "Warning: File size " << file_size << " doesn't match expected Standard (" 
                  << expected_size_standard << ") or Simple (" << expected_size_simple << ") format." << std::endl;
    }
    
    char* ptr = reinterpret_cast<char*>(data.data());
    size_t total_bytes = n * 8;
    size_t bytes_read = 0;
    const size_t CHUNK_SIZE = 1024 * 1024 * 1024; // 1GB chunks
    
    while (bytes_read < total_bytes) {
        size_t to_read = std::min(CHUNK_SIZE, total_bytes - bytes_read);
        in.read(ptr + bytes_read, to_read);
        size_t read_this_time = in.gcount();
        bytes_read += read_this_time;
        
        if (read_this_time < to_read) {
            break; // EOF or error
        }
    }

    if (bytes_read != total_bytes) {
        std::cerr << "Warning: Read fewer bytes than expected. Expected " << total_bytes << ", got " << bytes_read << std::endl;
    }
    
    return {data, decimals};
}

// Common fixture
class GefBenchmarkFixture : public benchmark::Fixture {
public:
    std::vector<int64_t> input_data;
    uint64_t universe;
    std::string current_basename;

    void SetUp(::benchmark::State& state) override {
        const size_t file_idx = static_cast<size_t>(state.range(0));
        if (file_idx >= g_input_files.size()) {
            state.SkipWithError("File index out of bounds.");
            return;
        }
        const auto& path = g_input_files[file_idx];
        current_basename = std::filesystem::path(path).filename().string();
        
        try {
            auto dataset = load_custom_dataset(path);
            input_data = std::move(dataset.data);
            universe = dataset.decimals;
        } catch (const std::exception& e) {
            state.SkipWithError(e.what());
            return;
        }

        if (input_data.empty()) {
            state.SkipWithError("Input data is empty.");
        }
    }
};
