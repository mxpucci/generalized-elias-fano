#ifndef GEF_HALF_BLOCK_RANK_HPP
#define GEF_HALF_BLOCK_RANK_HPP

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace gef::internal {

/**
 * rank1 over an LSB-first bitvector of at most MAX_BITS bits.
 *
 * The index is one uint16_t per 512-bit block: the number of ones before the
 * block's middle (bit 256), i.e. the density of Pasta's flat-rank directory
 * without its 44-bit base. A query reads only the 256-bit half holding its
 * position: ones in [256, i) of the block are added to the middle count, ones
 * in [i, 256) subtracted. It has no word loop and no data-dependent branch.
 *
 * The owner keeps the words alive and unmodified while the index is in use;
 * a copy of the owner must build its own index.
 */
class HalfBlockRank {
public:
    static constexpr size_t MAX_BITS = 65535;

    HalfBlockRank() = default;
    HalfBlockRank(const HalfBlockRank&) = delete;
    HalfBlockRank& operator=(const HalfBlockRank&) = delete;
    HalfBlockRank(HalfBlockRank&& other) noexcept { *this = std::move(other); }
    HalfBlockRank& operator=(HalfBlockRank&& other) noexcept {
        if (this != &other) {
            // The array moves with its buffer, so the views stay valid.
            middles_ = std::move(other.middles_);
            words_ = other.words_;
            num_words_ = other.num_words_;
            num_blocks_ = other.num_blocks_;
            other.clear();
        }
        return *this;
    }

    static bool supports(size_t bits) { return bits <= MAX_BITS; }

    void build(const uint64_t* words, size_t bits) {
        words_ = words;
        num_words_ = static_cast<uint32_t>((bits + 63) / 64);
        // rank1(bits) is valid, so the block of position `bits` is indexed too.
        num_blocks_ = static_cast<uint32_t>(bits / BLOCK_BITS + 1);
        middles_ = std::make_unique<uint16_t[]>(num_blocks_);
        size_t ones = 0, word = 0;
        for (size_t block = 0; block < num_blocks_; ++block) {
            for (const size_t middle = std::min<size_t>(block * 8 + HALF_WORDS, num_words_); word < middle; ++word)
                ones += std::popcount(words[word]);
            middles_[block] = static_cast<uint16_t>(ones);
        }
    }

    void clear() {
        middles_.reset();
        words_ = nullptr;
        num_words_ = num_blocks_ = 0;
    }

    [[nodiscard]] bool enabled() const { return num_blocks_ != 0; }

    // Ones in [0, index) for index <= bits.
    [[nodiscard]] size_t rank1(size_t index) const {
        // Halves are consecutive 256-bit units: half = index / 256 starts at
        // word 4 * half and is the upper one of its block when odd. Arithmetic
        // rather than conditional selections: the half is random per query.
        const size_t half = index / (BLOCK_BITS / 2);
        const size_t block = half / 2;
        const size_t upper = half & 1;
        const size_t first = half * HALF_WORDS;
        const size_t target = index % (BLOCK_BITS / 2); // position within the half
        const uint64_t* words = words_ + first;
        uint64_t tail[HALF_WORDS];
        if (first + HALF_WORDS > num_words_) [[unlikely]]
            words = copy_tail(first, tail);
        // Ones in [0, target) of the half: whole words before word target/64
        // (from prefix sums of the four popcounts) plus that word's low bits.
        // The lower half needs [target, 256) = total - [0, target).
        const size_t p0 = std::popcount(words[0]), p1 = std::popcount(words[1]);
        const size_t p2 = std::popcount(words[2]), p3 = std::popcount(words[3]);
        const size_t prefix[HALF_WORDS] = {0, p0, p0 + p1, p0 + p1 + p2};
        const size_t total = prefix[3] + p3;
        const size_t word = target / 64;
        const size_t below = prefix[word] +
            std::popcount(words[word] & ((uint64_t(1) << (target % 64)) - 1));
        return middles_[block] + below - (total & (upper - 1));
    }

    // The index entries plus the word and block counts.
    [[nodiscard]] size_t size_in_bytes() const {
        return enabled() ? num_blocks_ * sizeof(uint16_t) + 2 * sizeof(uint32_t) : 0;
    }

    [[nodiscard]] size_t allocated_size_in_bytes() const { return num_blocks_ * sizeof(uint16_t); }

private:
    static constexpr size_t BLOCK_BITS = 512;
    static constexpr size_t HALF_WORDS = 4;

    std::unique_ptr<uint16_t[]> middles_;
    const uint64_t* words_ = nullptr;
    uint32_t num_words_ = 0;  // at most MAX_BITS / 64 + 1
    uint32_t num_blocks_ = 0;

    // A final partial half: words past the vector count as zero. Out of line
    // to keep the query path small.
    [[gnu::cold, gnu::noinline]] const uint64_t* copy_tail(size_t first, uint64_t* tail) const {
        for (size_t k = 0; k < HALF_WORDS; ++k)
            tail[k] = first + k < num_words_ ? words_[first + k] : 0;
        return tail;
    }
};

} // namespace gef::internal

#endif
