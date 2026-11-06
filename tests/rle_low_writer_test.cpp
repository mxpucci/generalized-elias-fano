#include <gtest/gtest.h>
#include "gef/RLE_GEF.hpp"
#include <filesystem>
#include <limits>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

namespace {
template<typename T, typename Backend = PastaRankBitVector, bool RandomAccess = true>
void check_low_writer() {
    using U = std::make_unsigned_t<T>;
    using Codec = gef::internal::RLE_GEF<T, Backend, RandomAccess>;
    constexpr unsigned width = sizeof(T) * 8;
    std::mt19937_64 rng(187326);
    const T base = std::is_signed_v<T> ? std::numeric_limits<T>::min() : T(0);
    const auto path = std::filesystem::temp_directory_path() / "gef-rle-low-writer.bin";
    for (unsigned split = 0; split <= width; ++split) {
        const U low_mask = split == width ? U(~U(0)) : U((U(1) << split) - 1);
        for (size_t length : {size_t(0), size_t(1), size_t(2), size_t(7), size_t(63),
                              size_t(64), size_t(65), size_t(127), size_t(131)}) {
            std::vector<T> input(length);
            // Includes repeated highs, changing highs and full-domain residuals.
            for (size_t i = 0; i < length; ++i) {
                U residual = U(rng());
                if (i % 3 == 0) residual = 0;
                if (i % 7 == 0) residual = U(~U(0));
                input[i] = static_cast<T>(U(base) + residual);
            }
            Codec codec(input, gef::BlockParameters<T>{base, uint8_t(width), uint8_t(split)});
            sdsl::int_vector<> expected;
            if (split && length) {
                expected = sdsl::int_vector<>(length, 0, split);
                for (size_t i = 0; i < length; ++i)
                    expected[i] = U(U(input[i]) - U(base)) & low_mask;
            }
            EXPECT_EQ(codec.L.bit_size(), expected.bit_size());
            for (size_t i = 0; i < (expected.bit_size() + 63) / 64; ++i)
                EXPECT_EQ(codec.L.data()[i], expected.data()[i]) << split << ',' << length << ',' << i;
            std::vector<T> output(length);
            ASSERT_EQ(codec.get_elements(0, length, output), length);
            EXPECT_EQ(output, input) << split << ',' << length;
            if constexpr (RandomAccess)
                for (size_t i = 0; i < length; ++i) ASSERT_EQ(codec[i], input[i]);
            if (length == 65) {
                codec.serialize(path);
                Codec restored;
                restored.load(path);
                ASSERT_EQ(restored.get_elements(0, length, output), length);
                EXPECT_EQ(output, input) << split;
                std::filesystem::remove(path);
            }
        }
    }
}
}

TEST(RLELowWriter, SignedAndUnsignedIntegralValues) {
    check_low_writer<int8_t>();
    check_low_writer<uint8_t>();
    check_low_writer<int16_t>();
    check_low_writer<uint16_t>();
    check_low_writer<int32_t>();
    check_low_writer<uint32_t>();
    check_low_writer<int64_t>();
    check_low_writer<uint64_t>();
}

TEST(RLELowWriter, SDSLAndSupportFreeBackends) {
    check_low_writer<int64_t, SDSLBitVector>();
    check_low_writer<uint64_t, PastaRankBitVector, false>();
}
