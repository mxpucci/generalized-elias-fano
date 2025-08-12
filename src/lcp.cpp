#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include "gef/RLE_GEF.hpp"
#include "gef/U_GEF.hpp"
#include "gef/B_GEF_NO_RLE.hpp"
#include "datastructures/SDSLBitVectorFactory.hpp"
#include "gef/UniformedPartitioner.hpp"
#include "gef/utils.hpp"
// #include "sdsl/suffix_arrays.hpp" // sdsl::construct_sa, int_vector
#include "sdsl/qsufsort.hpp"
#include "datastructures/IBitVectorFactory.hpp"
#include "gef/IGEF.hpp"
#include <divsufsort.h>
#include <divsufsort64.h>

static sdsl::int_vector<64> sa_divsufsort_from_text_with_sentinel(const std::string& text) {
    const size_t n = text.size() + 1;
    std::vector<uint8_t> s(n);
    std::memcpy(s.data(), text.data(), text.size());
    s[n - 1] = 0;

    std::vector<saidx64_t> sa(n);
    int rc = divsufsort64(reinterpret_cast<const unsigned char*>(s.data()), sa.data(),
                          static_cast<saidx64_t>(n));
    if (rc != 0) {
        throw std::runtime_error("divsufsort64 failed with code " + std::to_string(rc));
    }

    sdsl::int_vector<64> sa_iv(n);
    for (size_t i = 0; i < n; ++i) sa_iv[i] = static_cast<uint64_t>(sa[i]);
    return sa_iv;
}

template<typename T>
struct U_GEF_Wrapper : public gef::U_GEF<T> {
    U_GEF_Wrapper(const std::vector<T>& data, std::shared_ptr<IBitVectorFactory> factory)
            : gef::U_GEF<T>(factory, data, gef::SplitPointStrategy::BRUTE_FORCE_SPLIT_POINT) {}
    U_GEF_Wrapper() : gef::U_GEF<T>() {}
};

template<typename T>
struct B_GEF_NO_RLE_Wrapper : public gef::B_GEF_NO_RLE<T> {
    B_GEF_NO_RLE_Wrapper(const std::vector<T>& data, std::shared_ptr<IBitVectorFactory> factory)
            : gef::B_GEF_NO_RLE<T>(factory, data, gef::SplitPointStrategy::BRUTE_FORCE_SPLIT_POINT) {}
    B_GEF_NO_RLE_Wrapper() : gef::B_GEF_NO_RLE<T>() {}
};

template<typename T>
struct RLE_GEF_Wrapper : public gef::RLE_GEF<T> {
    RLE_GEF_Wrapper(const std::vector<T>& data, std::shared_ptr<IBitVectorFactory> factory)
            : gef::RLE_GEF<T>(factory, data) {}
    RLE_GEF_Wrapper() : gef::RLE_GEF<T>() {}
};

static std::string read_file_to_string(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open file: " + filename);
    std::string contents;
    ifs.seekg(0, std::ios::end);
    contents.resize((size_t)ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    ifs.read(&contents[0], contents.size());
    return contents;
}

static std::vector<uint64_t> kasai_lcp_from_sa(const sdsl::int_vector<8>& text_with_sentinel,
                                               const sdsl::int_vector<64>& sa) {
    const size_t n = text_with_sentinel.size();
    std::vector<size_t> rank(n);
    for (size_t i = 0; i < n; ++i) {
        rank[(size_t)sa[i]] = i;
    }
    std::vector<uint64_t> lcp(n, 0);
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t r = rank[i];
        if (r + 1 == n) {
            lcp[r] = 0;
            k = 0;
            continue;
        }
        size_t j = (size_t)sa[r + 1];
        while (i + k < n && j + k < n &&
               text_with_sentinel[i + k] == text_with_sentinel[j + k]) {
            ++k;
        }
        lcp[r] = k;
        if (k > 0) --k;
    }
    return lcp;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <text-file> [..more files]\n";
        return 1;
    }

    for (int ai = 1; ai < argc; ++ai) {
        const std::string filename = argv[ai];
        std::cout << "=== File: " << filename << " ===\n";

        std::string text;
        try {
            text = read_file_to_string(filename);
        } catch (const std::exception& e) {
            std::cerr << "Error reading file: " << e.what() << "\n";
            continue;
        }

        sdsl::int_vector<8> text_iv(text.length() + 1);
        std::copy(text.begin(), text.end(), text_iv.begin());
        text_iv[text.length()] = 0;

        sdsl::int_vector<64> sa = sa_divsufsort_from_text_with_sentinel(text);

        std::vector<uint64_t> lcp_vec = kasai_lcp_from_sa(text_iv, sa);

        uint64_t max_lcp = 0;
        for (auto v : lcp_vec) max_lcp = std::max(max_lcp, v);
        uint8_t lcp_width = std::max<uint8_t>(1, sdsl::bits::hi(max_lcp) + 1);

        sdsl::int_vector<> lcp_iv(lcp_vec.size(), 0, lcp_width);
        for (size_t i = 0; i < lcp_vec.size(); ++i) lcp_iv[i] = lcp_vec[i];

        auto factory = std::make_shared<SDSLBitVectorFactory>();

        // Costruisci e stampa dimensioni per B_GEF_NO_RLE
        B_GEF_NO_RLE_Wrapper<uint64_t> gef_b_no_rle(lcp_vec, factory);

        // Costruisci e stampa dimensioni per U_GEF
        U_GEF_Wrapper<uint64_t> gef_u(lcp_vec, factory);

        // Costruisci e stampa dimensioni per RLE_GEF
        RLE_GEF_Wrapper<uint64_t> gef_rle(lcp_vec, factory);

        std::cout << "\n=== " << filename << " ===\n";
        std::cout << "lcp (raw): \t\t" << sdsl::size_in_bytes(lcp_iv) << " bytes\n";
        std::cout << "B_GEF_NO_RLE: \t\t" << gef_b_no_rle.size_in_bytes() << " bytes\n";
        std::cout << "U_GEF: \t\t\t" << gef_u.size_in_bytes() << " bytes\n";
        std::cout << "RLE_GEF: \t\t" << gef_rle.size_in_bytes() << " bytes\n";
        std::cout << "=====================================\n";
    }

    return 0;
}
