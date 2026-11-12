#!/usr/bin/env python3
"""Generates the floating-point forks RLE_GEF_FP.hpp and B_STAR_GEF_FP.hpp from the
integer headers RLE_GEF.hpp and B_STAR_GEF.hpp (which are not modified).

Each fork is a renamed copy of its integer class plus double-output copies of the
decoding kernels and of the helpers that hold the final values in registers: every
integer store of a decoded value becomes a store of its correctly rounded double
(FpConvert.hpp: fp::store_i64x8 / store_i64x4 / one), so a value is converted in
registers and written once. Low parts that the integer kernels unpack into the
output first are unpacked into a 64-value L1 block instead.

Every textual edit must match exactly once, so the script fails loudly when the
integer headers change shape; regenerate the forks after editing the integer
decoders.

usage: scripts/generate_fp_forks.py [include/gef]
"""
import os
import re
import sys

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "include", "gef")


def read(name):
    with open(os.path.join(DIR, name)) as f:
        return f.read()


def write(name, text):
    with open(os.path.join(DIR, name), "w") as f:
        f.write(text)
    print(f"wrote {name} ({len(text.splitlines())} lines)")


def rep(text, old, new, count=1):
    found = text.count(old)
    assert found == count, f"expected {count} match(es), found {found}:\n{old}"
    return text.replace(old, new)


def between(text, start, end):
    """Returns (index of start, index of end) with start included and end excluded."""
    a = text.index(start)
    b = text.index(end, a)
    return a, b


def no_integer_output(code, what):
    code = re.sub(r"//[^\n]*", "", code)  # comments may say "output"
    left = re.findall(r"(?<![\w])output(?![\w])", code)
    assert not left, f"{what}: integer output pointer still referenced"


FP_PREFETCH = """
#if GEF_FP_PREFETCH > 0
                // Streaming outputs (Scale::stream): request the output lines
                // GEF_FP_PREFETCH blocks ahead of the stores.
                if (k.stream && position + (GEF_FP_PREFETCH + 1) * 64 <= count) {
                    const char* ahead = reinterpret_cast<const char*>(block_output + GEF_FP_PREFETCH * 64);
                    for (int line = 0; line < 8; ++line) __builtin_prefetch(ahead + line * 64, 1, 3);
                }
#endif"""


def fallback_and_api(cls, worker_call_full):
    """Double-output worker, dispatch table and public accessor shared by both forks."""
    return f"""
        template<uint8_t SPLIT_POINT>
        size_t get_elements_fp_worker(size_t startIndex, size_t count, double* out, const fp::Scale& k) const {{
            if (count == 0 || startIndex >= m_num_elements) [[unlikely]] return 0;
            const size_t actual = std::min(count, m_num_elements - startIndex);
{worker_call_full}
            // Remaining cases: decode integers in L1-sized chunks (16 KB) and convert them.
            alignas(64) T buffer[2048];
            for (size_t done = 0; done < actual; done += 2048) {{
                const size_t len = std::min<size_t>(2048, actual - done);
                get_elements_worker<SPLIT_POINT>(startIndex + done, len, buffer);
                fp::scaled_to_double(buffer, out + done, len, k);
            }}
            return actual;
        }}

        template <size_t... Is>
        size_t dispatch_fp_worker(size_t b_val, size_t start, size_t count, double* out, const fp::Scale& k,
                                  std::index_sequence<Is...>) const {{
            using WorkerPtr = size_t ({cls}::*)(size_t, size_t, double*, const fp::Scale&) const;
            static constexpr WorkerPtr table[] = {{ &{cls}::get_elements_fp_worker<Is>... }};
            if (b_val >= sizeof...(Is)) throw std::invalid_argument("Invalid b value");
            return (this->*table[b_val])(start, count, out, k);
        }}
"""


# ---------------------------------------------------------------- RLE-GEF fork
def rle_fork():
    s = re.sub(r"\bRLE_GEF\b", "RLE_GEF_FP", read("RLE_GEF.hpp"))
    s = rep(s, "#ifndef RLE_GEF_H\n#define RLE_GEF_H\n", "#ifndef RLE_GEF_FP_H\n#define RLE_GEF_FP_H\n")
    s = rep(s, '#include "IGEF.hpp"', '#include "IGEF.hpp"\n#include "FpConvert.hpp"')

    # Helpers holding final values in registers: window4/8, fused block, block-by-bytes kernel.
    a, b = between(s, "#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)\n        template<uint8_t SPLIT_POINT, size_t INDEX>\n        static void reconstruct_window4(",
                   "        template<uint8_t SPLIT_POINT, bool Partial = false>\n        void reconstruct_marker_words(")
    h = s[a:b]
    m = h.index("        static __m512i map_high_byte(")
    h = h[:m] + h[h.index("\n        }\n", m) + len("\n        }\n"):]  # shared with the integer kernels
    h = rep(h, "static void reconstruct_window4(const uint64_t* words, __m256i highs, T* output) {",
            "static void reconstruct_window4_fp(const uint64_t* words, __m256i highs, double* output, const fp::Scale& k) {")
    h = rep(h, """            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + INDEX),
                               _mm256_add_epi64(highs, lows));""",
            "            fp::store_i64x4(output + INDEX, _mm256_add_epi64(highs, lows), k);")
    h = rep(h, """static void reconstruct_window4(const uint64_t* words, const uint64_t* bases,
                                        const uint8_t* ranks, T* output) {""",
            """static void reconstruct_window4_fp(const uint64_t* words, const uint64_t* bases,
                                        const uint8_t* ranks, double* output, const fp::Scale& k) {""")
    h = rep(h, "reconstruct_window4<SPLIT_POINT, INDEX>(words, highs, output);",
            "reconstruct_window4_fp<SPLIT_POINT, INDEX>(words, highs, output, k);")
    h = rep(h, "static void reconstruct_window8(const uint64_t* words, __m512i highs, T* output) {",
            "static void reconstruct_window8_fp(const uint64_t* words, __m512i highs, double* output, const fp::Scale& k) {")
    h = rep(h, "            _mm512_storeu_si512(output + INDEX, _mm512_add_epi64(highs, lows));",
            "            fp::store_i64x8(output + INDEX, _mm512_add_epi64(highs, lows), k);")
    h = rep(h, """static void reconstruct_fused_block(const uint64_t* words, uint64_t markers,
                                            const uint64_t* bases, T* output,""",
            """static void reconstruct_fused_block_fp(const uint64_t* words, uint64_t markers,
                                            const uint64_t* bases, double* output, const fp::Scale& k,""")
    h = rep(h, "reconstruct_window8<SPLIT_POINT, BYTES * 8>(words, highs, output);",
            "reconstruct_window8_fp<SPLIT_POINT, BYTES * 8>(words, highs, output, k);")
    h = rep(h, "reconstruct_window4<SPLIT_POINT, BYTES * 8", "reconstruct_window4_fp<SPLIT_POINT, BYTES * 8", count=2)
    h = rep(h, "ranks.data(), output);", "ranks.data(), output, k);")
    h = rep(h, "ranks.data() + 4, output);", "ranks.data() + 4, output, k);")
    h = rep(h, "static void reconstruct_block_by_bytes(", "static void reconstruct_block_by_bytes_fp(")
    h = rep(h, "uint64_t markers, size_t runs, T* output,\n",
            "uint64_t markers, size_t runs, const T* lows, double* output, const fp::Scale& k,\n")
    h = rep(h, "reconstruct_window8<SPLIT_POINT, BYTES * 8>(low_words, highs, output);",
            "reconstruct_window8_fp<SPLIT_POINT, BYTES * 8>(low_words, highs, output, k);")
    h = rep(h, """                    // Wider lows were unpacked into the output beforehand.
                    T* values = output + BYTES * 8;
                    if constexpr (SPLIT_POINT > 0)
                        highs = _mm512_add_epi64(highs, _mm512_loadu_si512(values));
                    _mm512_storeu_si512(values, highs);""",
            """                    // Wider lows were unpacked into `lows` beforehand.
                    if constexpr (SPLIT_POINT > 0)
                        highs = _mm512_add_epi64(highs, _mm512_loadu_si512(lows + BYTES * 8));
                    fp::store_i64x8(output + BYTES * 8, highs, k);""")
    h = "        // ---- Double-output copies of the helpers above (generated, see scripts/generate_fp_forks.py).\n" + h

    # Kernel: full partitions and partial ranges, writing doubles.
    ka, kb = between(s, "        template<uint8_t SPLIT_POINT, bool Partial = false>\n        void reconstruct_marker_words(",
                     "        template<uint8_t SPLIT_POINT>\n        [[gnu::flatten]] size_t get_elements_worker")
    k = s[ka:kb]
    k = rep(k, "void reconstruct_marker_words(size_t count, T* output, size_t startIndex = 0) const {",
            "void reconstruct_marker_words_fp(size_t count, double* doutput, const fp::Scale& k_in, size_t startIndex = 0) const {\n"
            "            const fp::Scale k = k_in; // local copy: stays in registers, output stores cannot alias it\n"
            "            alignas(64) T blk[64]; // low parts of one block (L1)")
    k = rep(k, "const auto decode_block = [&](size_t block, size_t runs_before_block, T* dst) {",
            "const auto decode_block = [&](size_t block, size_t runs_before_block, double* ddst) {")
    k = rep(k, "unpack_aligned_block<SPLIT_POINT>(low_block, dst, 0, std::make_index_sequence<64>{});",
            "unpack_aligned_block<SPLIT_POINT>(low_block, blk, 0, std::make_index_sequence<64>{});")
    k = rep(k, "reconstruct_block_by_bytes<SPLIT_POINT, true>(", "reconstruct_block_by_bytes_fp<SPLIT_POINT, true>(")
    k = rep(k, "reconstruct_block_by_bytes<SPLIT_POINT, false>(", "reconstruct_block_by_bytes_fp<SPLIT_POINT, false>(")
    k = rep(k, "byte_base, bits[block >> 6], runs_before_block, dst, std::make_index_sequence<8>{});",
            "byte_base, bits[block >> 6], runs_before_block, blk, ddst, k, std::make_index_sequence<8>{});", count=2)
    k = rep(k, "const auto decode_block = [](size_t, size_t, T*) {};", "const auto decode_block = [](size_t, size_t, double*) {};")
    k = rep(k, """                            alignas(64) T scratch[64];
                            decode_block(block, block_runs, scratch);
                            const size_t end = std::min(block + 64, count);
                            std::copy(scratch + (position & 63), scratch + (end - block), output);""",
            """                            alignas(64) double scratch[64];
                            decode_block(block, block_runs, scratch);
                            const size_t end = std::min(block + 64, count);
                            std::copy(scratch + (position & 63), scratch + (end - block), doutput);""")
    k = rep(k, """            if constexpr (SPLIT_POINT > 0 && !Partial)
                unpack_low_range<SPLIT_POINT>(position, count, output + position, 0);
            uint64_t carry = carried();
            for (; position < count; ++position) {
                if ((bits[position >> 6] >> (position & 63)) & 1)
                    carry = run_value(runs++);
                if constexpr (Partial)
                    output[position - begin] = static_cast<T>(carry +
                        unpack_low<SPLIT_POINT>(L.data(), position));
                else if constexpr (SPLIT_POINT == 0)
                    output[position] = static_cast<T>(carry);
                else
                    output[position] = static_cast<T>(carry + static_cast<U>(output[position]));
            }""", """            const size_t tail_start = position; // fewer than 64 values remain
            if constexpr (SPLIT_POINT > 0 && !Partial)
                if (position < count) unpack_low_range<SPLIT_POINT>(position, count, blk, 0);
            uint64_t carry = carried();
            for (; position < count; ++position) {
                if ((bits[position >> 6] >> (position & 63)) & 1)
                    carry = run_value(runs++);
                if constexpr (Partial)
                    doutput[position - begin] = fp::one(static_cast<T>(carry +
                        unpack_low<SPLIT_POINT>(L.data(), position)), k);
                else if constexpr (SPLIT_POINT == 0)
                    doutput[position] = fp::one(static_cast<T>(carry), k);
                else
                    doutput[position] = fp::one(static_cast<T>(carry + static_cast<U>(blk[position - tail_start])), k);
            }""")
    k = rep(k, """                    output[position - begin] = static_cast<T>(carry +
                        unpack_low<SPLIT_POINT>(L.data(), position));""",
            """                    doutput[position - begin] = fp::one(static_cast<T>(carry +
                        unpack_low<SPLIT_POINT>(L.data(), position)), k);""")
    k = rep(k, "                T* block_output = output + (position - begin);",
            "                double* block_output = doutput + (position - begin);" + FP_PREFETCH)
    k = rep(k, "reconstruct_fused_block<SPLIT_POINT>(", "reconstruct_fused_block_fp<SPLIT_POINT>(")
    k = rep(k, "run_bases.data(), block_output, std::make_index_sequence<8>{});",
            "run_bases.data(), block_output, k, std::make_index_sequence<8>{});")
    k = rep(k, """                        block_output, 0, std::make_index_sequence<64>{});""",
            """                        blk, 0, std::make_index_sequence<64>{});""")
    k = rep(k, "                    T* dst = block_output + byte * 8;",
            "                    T* dst = blk + byte * 8;\n                    double* ddst = block_output + byte * 8;")
    k = rep(k, "                        _mm512_storeu_si512(dst, values);", "                        fp::store_i64x8(ddst, values, k);")
    k = rep(k, "                                dst[lane] = static_cast<T>(value);",
            "                                ddst[lane] = fp::one(static_cast<T>(value), k);")
    k = rep(k, "                                dst[lane] = static_cast<T>(value + static_cast<U>(dst[lane]));",
            "                                ddst[lane] = fp::one(static_cast<T>(value + static_cast<U>(dst[lane])), k);")
    k = rep(k, """                        alignas(64) T scratch[64];
                        decode_block(position, runs, scratch);
                        std::copy(scratch, scratch + (count - position), output + (position - begin));""",
            """                        alignas(64) double scratch[64];
                        decode_block(position, runs, scratch);
                        std::copy(scratch, scratch + (count - position), doutput + (position - begin));""")
    no_integer_output(k, "RLE kernel")
    k = ("        // Double-output copy of reconstruct_marker_words (generated): decoded values are converted\n"
         "        // in registers (fp::store_i64x8 / store_i64x4 / one) and stored once, as doubles.\n") + k

    worker = """            if (h != 0) {
                if constexpr (!ExceptionBitVectorType::reverse_bit_order) {
                    if (startIndex == 0 && actual == m_num_elements && actual >= 64) {
                        reconstruct_marker_words_fp<SPLIT_POINT>(actual, out, k);
                        return actual;
                    }
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD)
                    if constexpr (sizeof(T) == 8 && SPLIT_POINT < 64) {
                        if (actual >= 64 && ((64 - (startIndex & 63)) & 63) <= actual - 64) {
                            reconstruct_marker_words_fp<SPLIT_POINT, true>(actual, out, k, startIndex);
                            return actual;
                        }
                    }
#endif
                }
            }"""
    api = fallback_and_api("RLE_GEF_FP", worker) + """
    public:
        // Decodes [startIndex, startIndex + count) as correctly rounded doubles v / scale
        // (inv = 1 / scale), converting in registers before the only store of each value;
        // `stream` marks outputs too large for the caches (see GEF_FP_PREFETCH_MIN_VALUES);
        // `two_op` selects the cheaper conversion, which callers must have verified (FpConvert.hpp).
        size_t get_elements_into(size_t startIndex, size_t count, double* output, double scale, double inv,
                                 bool stream = false, bool two_op = false) const {
            if (count == 0 || startIndex >= m_num_elements) return 0;
            const size_t actual = std::min(count, m_num_elements - startIndex);
            const fp::Scale k(scale, inv, stream, two_op);
            if constexpr (RandomAccess) {
                return dispatch_fp_worker(b, startIndex, actual, output, k, std::make_index_sequence<sizeof(T) * 8 + 1>{});
            } else {
                std::vector<T> tmp(actual);
                get_elements_into(startIndex, actual, tmp.data());
                fp::scaled_to_double(tmp.data(), output, actual, k);
                return actual;
            }
        }
    private:
"""
    # Insert the copies after the original kernel; the integer paths still use the originals.
    s = s[:kb] + h + "\n" + k + api + s[kb:]
    write("RLE_GEF_FP.hpp", s)


# ---------------------------------------------------------------- B*-GEF fork
def bstar_fork():
    s = re.sub(r"\bB_STAR_GEF\b", "B_STAR_GEF_FP", read("B_STAR_GEF.hpp"))
    s = rep(s, "#ifndef B_STAR_GEF_HPP\n#define B_STAR_GEF_HPP", "#ifndef B_STAR_GEF_FP_HPP\n#define B_STAR_GEF_FP_HPP")
    # Namespace-scope helpers exist in the integer header: include it and reuse them.
    a = s.index("    struct BStarZeroByte {")
    b = s.index("    template<typename T, typename GapBitVectorType", a)
    s = s[:a] + s[b:]
    s = rep(s, '#include "IGEF.hpp"', '#include "IGEF.hpp"\n#include "FpConvert.hpp"\n#include "B_STAR_GEF.hpp"')

    ha, hb = between(s, "#if defined(__AVX2__) && defined(__AVX512F__) && !defined(GEF_DISABLE_SIMD) && SIZE_MAX == UINT64_MAX\n        template<uint8_t SPLIT_POINT, size_t INDEX>\n        static void reconstruct_low8(",
                     "        template<uint8_t SPLIT_POINT, typename Position>\n        size_t decode_position_ranges(")
    h = s[ha:hb]
    h = rep(h, """static void reconstruct_low8(const uint64_t* words, const uint32_t* plus_positions,
                                     const uint32_t* minus_positions, __m512i base_vector, T* output) {""",
            """static void reconstruct_low8_fp(const uint64_t* words, const uint32_t* plus_positions,
                                     const uint32_t* minus_positions, __m512i base_vector, double* output,
                                     const fp::Scale& k) {""")
    h = rep(h, """            _mm512_storeu_si512(output + INDEX,
                _mm512_add_epi64(base_vector, _mm512_or_si512(high, lows)));""",
            "            fp::store_i64x8(output + INDEX, _mm512_add_epi64(base_vector, _mm512_or_si512(high, lows)), k);")
    h = rep(h, "static void reconstruct_aligned64(", "static void reconstruct_aligned64_fp(")
    h = rep(h, "                                          T* output,\n",
            "                                          double* output, const fp::Scale& k,\n")
    h = rep(h, "reconstruct_low8<SPLIT_POINT, I>(low_data, plus_positions, minus_positions, base_vector, output);",
            "reconstruct_low8_fp<SPLIT_POINT, I>(low_data, plus_positions, minus_positions, base_vector, output, k);")
    h = rep(h, "                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + I), values);",
            "                        fp::store_i64x4(output + I, values, k);")
    h = rep(h, "                    output[I] = static_cast<T>(base_u + (high | low));",
            "                    output[I] = fp::one(static_cast<T>(base_u + (high | low)), k);")
    h = "        // ---- Double-output copies of the helpers above (generated, see scripts/generate_fp_forks.py).\n" + h

    ka, kb = between(s, "        template<uint8_t SPLIT_POINT, typename Position>\n        size_t decode_position_ranges(",
                     "        template<uint8_t SPLIT_POINT>\n        size_t get_elements_worker(")
    k = s[ka:kb]
    k = rep(k, "size_t decode_position_ranges(size_t startIndex, size_t count, T* output) const {",
            "size_t decode_position_ranges_fp(size_t startIndex, size_t count, double* doutput, const fp::Scale& k_in) const {\n"
            "            const fp::Scale k = k_in; // local copy: stays in registers, output stores cannot alias it")
    k = rep(k, "            alignas(64) Position minus_positions[MAX_BATCH];",
            "            alignas(64) Position minus_positions[MAX_BATCH];\n"
            "            // Without the aligned SIMD path (split point 0 or 64) every value is reconstructed\n"
            "            // by the scalar loop: keep it integer (vectorizable) and convert the batch at once.\n"
            "            constexpr bool scalar_batches = !(SPLIT_POINT > 0 && SPLIT_POINT < sizeof(T) * 8);\n"
            "            alignas(64) T batch_values[scalar_batches ? MAX_BATCH : 1];")
    k = rep(k, "                    output[produced + i] = static_cast<T>(base_u + (high | low));",
            "                    if constexpr (scalar_batches) batch_values[i] = static_cast<T>(base_u + (high | low));\n"
            "                    else doutput[produced + i] = fp::one(static_cast<T>(base_u + (high | low)), k);")
    k = rep(k, "                for (; i < batch; ++i) reconstruct_scalar(i);\n                produced += batch;",
            "#if defined(GEF_FP_VECTOR) && defined(__AVX512F__)\n"
            "                if constexpr (SPLIT_POINT == 0 && sizeof(T) == 8) {\n"
            "                    // No low parts: value = base + plus - minus, eight at a time, stored as doubles.\n"
            "                    // Positions are bounded by the INT32 guard, so 32-bit differences cannot overflow.\n"
            "                    const __m512i base_vector = _mm512_set1_epi64(std::bit_cast<int64_t>(static_cast<uint64_t>(base_u)));\n"
            "                    for (; batch - i >= 8; i += 8) {\n"
            "                        __m512i difference;\n"
            "                        if constexpr (sizeof(Position) == 4)\n"
            "                            difference = _mm512_cvtepi32_epi64(_mm256_sub_epi32(\n"
            "                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(plus_positions + i)),\n"
            "                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(minus_positions + i))));\n"
            "                        else\n"
            "                            difference = _mm512_sub_epi64(_mm512_loadu_si512(plus_positions + i),\n"
            "                                                          _mm512_loadu_si512(minus_positions + i));\n"
            "                        fp::store_i64x8(doutput + produced + i, _mm512_add_epi64(base_vector, difference), k);\n"
            "                    }\n"
            "                }\n"
            "#endif\n"
            "                const size_t scalar_from = i;\n"
            "                for (; i < batch; ++i) reconstruct_scalar(i);\n"
            "                if constexpr (scalar_batches) {\n"
            "                    for (size_t done = scalar_from; done < batch; done += 64) {\n"
            "#if GEF_FP_PREFETCH > 0\n"
            "                        if (k.stream && produced + done + (GEF_FP_PREFETCH + 1) * 64 <= count) {\n"
            "                            const char* ahead = reinterpret_cast<const char*>(doutput + produced + done + GEF_FP_PREFETCH * 64);\n"
            "                            for (int line = 0; line < 8; ++line) __builtin_prefetch(ahead + line * 64, 1, 3);\n"
            "                        }\n"
            "#endif\n"
            "                        fp::scaled_to_double(batch_values + done, doutput + produced + done,\n"
            "                                             std::min<size_t>(64, batch - done), k);\n"
            "                    }\n"
            "                }\n"
            "                produced += batch;")
    k = rep(k, "reconstruct_aligned64<SPLIT_POINT>(block, plus_positions + i, minus_positions + i,",
            "reconstruct_aligned64_fp<SPLIT_POINT>(block, plus_positions + i, minus_positions + i,")
    k = rep(k, "                                                           base_u, output + produced + i,",
            "                                                           base_u, doutput + produced + i, k,")
    k = rep(k, "                    for (; batch - i >= 64; i += 64) {\n",
            """                    for (; batch - i >= 64; i += 64) {
#if GEF_FP_PREFETCH > 0
                        // Streaming outputs (Scale::stream): request output lines ahead.
                        if (k.stream && produced + i + (GEF_FP_PREFETCH + 1) * 64 <= count) {
                            const char* ahead = reinterpret_cast<const char*>(doutput + produced + i + GEF_FP_PREFETCH * 64);
                            for (int line = 0; line < 8; ++line) __builtin_prefetch(ahead + line * 64, 1, 3);
                        }
#endif
""")
    no_integer_output(k, "B* kernel")
    k = ("        // Double-output copy of decode_position_ranges (generated): values are converted in\n"
         "        // registers (fp::store_i64x8 / store_i64x4 / one) and stored once, as doubles.\n") + k

    worker = """            if (h != 0) {
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD) && SIZE_MAX == UINT64_MAX
                if constexpr (sizeof(T) == 8 && !GapBitVectorType::reverse_bit_order) {
                    if (G_plus->size() <= INT32_MAX && G_minus->size() <= INT32_MAX)
                        return decode_position_ranges_fp<SPLIT_POINT, uint32_t>(startIndex, actual, out, k);
                }
#endif
                return decode_position_ranges_fp<SPLIT_POINT, size_t>(startIndex, actual, out, k);
            }"""
    api = fallback_and_api("B_STAR_GEF_FP", worker) + """
    public:
        // Decodes [startIndex, startIndex + count) as correctly rounded doubles v / scale.
        size_t get_elements_into(size_t startIndex, size_t count, double* output, double scale, double inv,
                                 bool stream = false, bool two_op = false) const
            requires (RandomAccess) {
            const fp::Scale k(scale, inv, stream, two_op);
            return dispatch_fp_worker(b, startIndex, count, output, k, std::make_index_sequence<65>{});
        }
    private:
"""
    s = s[:kb] + h + "\n" + k + api + s[kb:]
    write("B_STAR_GEF_FP.hpp", s)


if __name__ == "__main__":
    rle_fork()
    bstar_fork()
