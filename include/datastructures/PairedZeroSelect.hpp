#ifndef GEF_PAIRED_ZERO_SELECT_HPP
#define GEF_PAIRED_ZERO_SELECT_HPP

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__BMI__) && defined(__BMI2__) && !defined(GEF_DISABLE_SIMD)
#include <immintrin.h>
#define GEF_PAIRED_ZERO_SELECT_AVX512 1
#endif

namespace gef::internal {

/**
 * select0 over two LSB-first bitvectors queried with the same zero rank, as
 * B* gap streams are: each stores exactly one zero per element.
 *
 * The only index is, per 512-bit block (eight payload words), the number of
 * zeros before that block as a uint16_t. That is Pasta's L1/L2 density (128
 * bits per 4096) without its select samples. A query compares the rank with
 * all block counts at once, then finds the zero among the block's eight words,
 * so it has no data-dependent branches and only two dependent index loads.
 *
 * Counts must fit uint16_t, so the directory requires at most MAX_ZEROS zeros
 * per vector; callers keep a general select structure for larger inputs.
 */
class PairedZeroSelect {
public:
    static constexpr size_t MAX_ZEROS = 65535;

    PairedZeroSelect() = default;
    PairedZeroSelect(const PairedZeroSelect&) = delete;
    PairedZeroSelect& operator=(const PairedZeroSelect&) = delete;
    PairedZeroSelect(PairedZeroSelect&& other) noexcept { *this = std::move(other); }
    PairedZeroSelect& operator=(PairedZeroSelect&& other) noexcept {
        if (this != &other) {
            // Moving the vector keeps its buffer, so the count views stay valid.
            counts_ = std::move(other.counts_);
            side_[0] = other.side_[0];
            side_[1] = other.side_[1];
            other.clear();
        }
        return *this;
    }

    static bool supports(size_t zeros, size_t plus_bits, size_t minus_bits) {
        return zeros <= MAX_ZEROS && plus_bits / 64 < UINT32_MAX && minus_bits / 64 < UINT32_MAX;
    }

    // Each vector holds `zeros` zeros (at most MAX_ZEROS, see supports()) and
    // must stay alive and unmodified while this directory is in use. Only the
    // words up to the last of them are indexed and ever read by queries.
    void build(const uint64_t* plus, size_t plus_bits, const uint64_t* minus, size_t minus_bits,
               size_t zeros) {
        const uint64_t* words[2] = {plus, minus};
        const size_t bits[2] = {plus_bits, minus_bits};
        size_t num_words[2];
        for (int s = 0; s < 2; ++s) {
            size_t w = 0;
            for (size_t seen = 0; w < (bits[s] + 63) / 64 && seen < zeros; ++w)
                seen += std::popcount(~words[s][w]);
            num_words[s] = w;
        }
        size_t entries[2];
        for (int s = 0; s < 2; ++s) {
            const size_t blocks = (num_words[s] + BLOCK_WORDS - 1) / BLOCK_WORDS;
            entries[s] = (blocks + CHUNK - 1) / CHUNK * CHUNK;
        }
        // Padding entries exceed every valid rank, so they are never counted.
        counts_.assign(entries[0] + entries[1], UINT16_MAX);
        counts_.shrink_to_fit();
        size_t offset = 0;
        for (int s = 0; s < 2; ++s) {
            uint16_t* counts = counts_.data() + offset;
            size_t zeros = 0;
            for (size_t w = 0; w < num_words[s]; ++w) {
                if (w % BLOCK_WORDS == 0) counts[w / BLOCK_WORDS] = static_cast<uint16_t>(zeros);
                zeros += std::popcount(~words[s][w]);
            }
            side_[s] = {words[s], counts, static_cast<uint32_t>(num_words[s]),
                        static_cast<uint32_t>(entries[s] / CHUNK)};
            offset += entries[s];
        }
    }

    void clear() {
        counts_.clear();
        counts_.shrink_to_fit();
        side_[0] = side_[1] = Side{};
    }

    [[nodiscard]] bool enabled() const { return !counts_.empty(); }

    // Positions of the rank-th zero (1-based) in the plus and minus vectors.
    [[gnu::always_inline]] inline std::pair<size_t, size_t> select0_pair(size_t rank) const {
#if defined(GEF_PAIRED_ZERO_SELECT_AVX512)
        return select0_pair_staged(rank);
#else
        return {select0_in(side_[0], rank), select0_in(side_[1], rank)};
#endif
    }

    [[nodiscard]] size_t select0(size_t side, size_t rank) const {
        return select0_in(side_[side], rank);
    }

    // Directory entries plus each side's word count and chunk count.
    [[nodiscard]] size_t size_in_bytes() const {
        return enabled() ? counts_.size() * sizeof(uint16_t) + 4 * sizeof(uint32_t) : 0;
    }

    [[nodiscard]] size_t allocated_size_in_bytes() const {
        return counts_.capacity() * sizeof(uint16_t);
    }

private:
    static constexpr size_t BLOCK_WORDS = 8;
    static constexpr size_t CHUNK = 32; // uint16_t counts per 512-bit compare

    struct Side {
        const uint64_t* words = nullptr;
        const uint16_t* counts = nullptr;
        uint32_t num_words = 0;
        uint32_t chunks = 0;
    };

    std::vector<uint16_t> counts_;
    Side side_[2];

#if defined(GEF_PAIRED_ZERO_SELECT_AVX512)
    // Block = (number of blocks with fewer than rank zeros before them) - 1.
    // The chunk count is fixed per vector, so this loop predicts well;
    // full loads avoid the cost of masked ones.
    [[gnu::always_inline]] static inline size_t find_block(const Side& s, __m512i target) {
        size_t below = 0;
        for (uint32_t c = 0; c < s.chunks; ++c) {
            const __m512i counts = _mm512_loadu_si512(s.counts + c * CHUNK);
            below += std::popcount(static_cast<uint32_t>(_mm512_cmplt_epu16_mask(counts, target)));
        }
        return below - 1;
    }

    // The block's eight words. Only a final partial block uses a masked
    // load; its absent words follow the answer, so they do not affect it.
    [[gnu::always_inline]] static inline __m512i load_block(const Side& s, size_t block) {
        const size_t first = block * BLOCK_WORDS;
        const size_t present = s.num_words - first;
        return present >= BLOCK_WORDS
            ? _mm512_loadu_si512(s.words + first)
            : _mm512_maskz_loadu_epi64(static_cast<__mmask8>((1u << present) - 1), s.words + first);
    }

    // Position of the in_block-th zero (1-based) among the block's words.
    [[gnu::always_inline]] static inline size_t select_in_block(const Side& s, size_t block, __m512i bits,
                                                                size_t in_block) {
        bits = _mm512_ternarylogic_epi64(bits, bits, bits, 0x55); // complement
        const __m512i nibble_counts = _mm512_broadcast_i32x4(
            _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4));
        const __m512i low_nibbles = _mm512_set1_epi8(0x0f);
        const __m512i zero = _mm512_setzero_si512();
        const __m512i byte_counts = _mm512_add_epi8(
            _mm512_shuffle_epi8(nibble_counts, _mm512_and_si512(bits, low_nibbles)),
            _mm512_shuffle_epi8(nibble_counts, _mm512_and_si512(_mm512_srli_epi16(bits, 4), low_nibbles)));
        __m512i inclusive = _mm512_sad_epu8(byte_counts, zero);
        inclusive = _mm512_add_epi64(inclusive, _mm512_alignr_epi64(inclusive, zero, 7));
        inclusive = _mm512_add_epi64(inclusive, _mm512_alignr_epi64(inclusive, zero, 6));
        inclusive = _mm512_add_epi64(inclusive, _mm512_alignr_epi64(inclusive, zero, 4));
        const __m512i exclusive = _mm512_alignr_epi64(inclusive, zero, 7);
        const unsigned word = std::popcount(static_cast<unsigned>(
            _mm512_cmplt_epu64_mask(inclusive, _mm512_set1_epi64(static_cast<long long>(in_block)))));
        const size_t before = static_cast<size_t>(_mm_cvtsi128_si64(_mm512_castsi512_si128(
            _mm512_permutexvar_epi64(_mm512_set1_epi64(word), exclusive))));
        const size_t position = block * BLOCK_WORDS + word;
        const uint64_t zeros = ~s.words[position];
        return position * 64 + _tzcnt_u64(_pdep_u64(uint64_t(1) << (in_block - before - 1), zeros));
    }

    // Both sides advance stage by stage, so each side's loads issue before
    // the other side's dependent computation fills the scheduler.
    [[gnu::always_inline]] inline std::pair<size_t, size_t> select0_pair_staged(size_t rank) const {
        const __m512i target = _mm512_set1_epi16(static_cast<short>(rank));
        const size_t block0 = find_block(side_[0], target);
        const size_t block1 = find_block(side_[1], target);
        const __m512i bits0 = load_block(side_[0], block0);
        const __m512i bits1 = load_block(side_[1], block1);
        const size_t in_block0 = rank - side_[0].counts[block0];
        const size_t in_block1 = rank - side_[1].counts[block1];
        return {select_in_block(side_[0], block0, bits0, in_block0),
                select_in_block(side_[1], block1, bits1, in_block1)};
    }
#endif

    [[gnu::always_inline]] static inline size_t select0_in(const Side& s, size_t rank) {
#if defined(GEF_PAIRED_ZERO_SELECT_AVX512)
        const size_t block = find_block(s, _mm512_set1_epi16(static_cast<short>(rank)));
        return select_in_block(s, block, load_block(s, block), rank - s.counts[block]);
#else
        // Last block whose preceding zero count is below rank (counts are sorted).
        const uint16_t* end = s.counts + static_cast<size_t>(s.chunks) * CHUNK;
        const size_t block = static_cast<size_t>(
            std::upper_bound(s.counts, end, static_cast<uint16_t>(rank - 1)) - s.counts) - 1;
        size_t remaining = rank - s.counts[block];
        size_t word = block * BLOCK_WORDS;
        for (;; ++word) {
            const size_t zeros = static_cast<size_t>(std::popcount(~s.words[word]));
            if (zeros >= remaining) break;
            remaining -= zeros;
        }
        uint64_t zeros = ~s.words[word];
        for (size_t i = 1; i < remaining; ++i) zeros &= zeros - 1;
        return word * 64 + static_cast<size_t>(std::countr_zero(zeros));
#endif
    }
};

} // namespace gef::internal

#endif
