#include <gtest/gtest.h>
#include "gef/FastBitWriter.hpp"
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {
// Appends to a plain bit list, lowest bit first.
struct ReferenceBits {
    std::vector<bool> bits;
    void append(uint64_t value, unsigned width) {
        for (unsigned i = 0; i < width; ++i) bits.push_back((value >> i) & 1);
    }
    void append_unary(uint64_t count) {
        bits.insert(bits.end(), count, true);
        bits.push_back(false);
    }
};

// Words beyond the last written bit must be left untouched.
void expect_same_words(const std::vector<uint64_t>& words, const ReferenceBits& reference,
                       uint64_t sentinel, const char* what) {
    const size_t used = (reference.bits.size() + 63) / 64;
    for (size_t w = 0; w < words.size(); ++w) {
        if (w >= used) {
            EXPECT_EQ(words[w], sentinel) << what << " wrote past the last field, word " << w;
            continue;
        }
        uint64_t expected = 0;
        for (size_t i = 0; i < 64 && w * 64 + i < reference.bits.size(); ++i)
            expected |= uint64_t(reference.bits[w * 64 + i]) << i;
        ASSERT_EQ(words[w], expected) << what << " word " << w;
    }
}
}

TEST(PackedBitWriter, FieldsOfEveryWidthIncludingZero) {
    std::mt19937_64 rng(4242);
    for (int round = 0; round < 200; ++round) {
        ReferenceBits reference;
        std::vector<uint64_t> words(257, ~0ULL);  // sentinel: all ones
        gef::PackedBitWriter writer(words.data());
        const size_t fields = rng() % 200;
        for (size_t f = 0; f < fields && reference.bits.size() + 64 <= (words.size() - 1) * 64; ++f) {
            const unsigned width = round < 65 ? round : rng() % 65;  // first rounds: one fixed width
            const uint64_t value = width == 64 ? rng() : rng() & ((uint64_t(1) << width) - 1);
            writer.append(value, width);
            reference.append(value, width);
        }
        writer.finish();
        // Bits past the end of the last field inside its word must be zero.
        expect_same_words(words, reference, ~0ULL, "append");
    }
}

TEST(PackedBitWriter, UnaryCodesShortAndLong) {
    std::mt19937_64 rng(77);
    const std::vector<uint64_t> fixed = {0, 1, 2, 62, 63, 64, 65, 127, 128, 129, 191, 192, 1000};
    for (int round = 0; round < 100; ++round) {
        ReferenceBits reference;
        std::vector<uint64_t> words(2048, 0x5555555555555555ULL);
        gef::PackedBitWriter writer(words.data());
        for (int k = 0; k < 40; ++k) {
            // Interleave plain fields so codes start at every bit offset.
            const unsigned width = rng() % 65;
            const uint64_t value = width == 64 ? rng() : rng() & ((uint64_t(1) << width) - 1);
            writer.append(value, width);
            reference.append(value, width);
            const uint64_t count = round < 13 ? fixed[round] : (rng() % 4 ? rng() % 70 : rng() % 400);
            writer.append_unary(count);
            reference.append_unary(count);
        }
        writer.finish();
        expect_same_words(words, reference, 0x5555555555555555ULL, "unary");
    }
}

