#!/usr/bin/env python3
"""Generates the floating-point forks RLE_GEF_FP.hpp and B_STAR_GEF_FP.hpp from the
integer headers RLE_GEF.hpp and B_STAR_GEF.hpp (which are not modified).

Each fork is a renamed copy of its integer class plus a double-output decoder:
the full-range kernel decodes into an L1 block and converts it (FpConvert.hpp)
before the only store of each value. Every textual edit must match exactly once,
so the script fails loudly if the integer headers change shape; regenerate the
forks after editing the integer decoders.

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


# ---------------------------------------------------------------- RLE-GEF fork
src = read("RLE_GEF.hpp")
s = re.sub(r"\bRLE_GEF\b", "RLE_GEF_FP", src).replace("RLE_GEF_H\n", "RLE_GEF_FP_H\n", 2)
s = s.replace("#include \"IGEF.hpp\"", "#include \"IGEF.hpp\"\n#include \"FpConvert.hpp\"", 1)
start = s.index("        template<uint8_t SPLIT_POINT, bool Partial = false>\n        void reconstruct_marker_words(")
end = s.index("        template<uint8_t SPLIT_POINT>\n        [[gnu::flatten]] size_t get_elements_worker", start)
k = s[start:end]
fp = k.replace("        template<uint8_t SPLIT_POINT, bool Partial = false>\n        void reconstruct_marker_words(size_t count, T* output, size_t startIndex = 0) const {",
"""        // Double-output copy of the full-partition kernel (Partial = false): each 64-value
        // block is decoded into an L1 scratch block and converted into `doutput`, so every
        // output value is stored once, as a double. scale = 10^p, inv = 1 / scale.
        template<uint8_t SPLIT_POINT>
        void reconstruct_marker_words_fp(size_t count, double* doutput, double scale, double inv) const {
            constexpr bool Partial = false;
            constexpr size_t startIndex = 0;
            alignas(64) T blk[64];
            T* output = blk; (void)output;""", 1)
assert fp != k
def rep(a, b):
    global fp
    assert fp.count(a) == 1, a
    fp = fp.replace(a, b)
rep("                T* block_output = output + (position - begin);", """                T* block_output = blk;
#if GEF_FP_PREFETCH > 0
                // The block is stored in one burst from L1: request its output lines
                // for writing GEF_FP_PREFETCH blocks ahead, as the integer kernel
                // implicitly does by writing low bits before reconstructing.
                if (position + (GEF_FP_PREFETCH + 1) * 64 <= count) {
                    const char* ahead = reinterpret_cast<const char*>(doutput + position + GEF_FP_PREFETCH * 64);
                    for (int line = 0; line < 8; ++line) __builtin_prefetch(ahead + line * 64, 1, 3);
                }
#endif""")
rep("""                        decode_block(position, runs, block_output);
                        runs += block_runs;
                        continue;""", """                        decode_block(position, runs, block_output);
                        runs += block_runs;
                        fp::scaled_to_double(blk, doutput + position, 64, scale, inv);
                        continue;""")
rep("""                        run_bases.data(), block_output, std::make_index_sequence<8>{});
                    continue;""", """                        run_bases.data(), block_output, std::make_index_sequence<8>{});
                    fp::scaled_to_double(blk, doutput + position, 64, scale, inv);
                    continue;""")
rep("""                    preceding += ranks[7];
                }
            }""", """                    preceding += ranks[7];
                }
                fp::scaled_to_double(blk, doutput + position, 64, scale, inv);
            }""")
tail_old = fp[fp.index("            if constexpr (SPLIT_POINT > 0 && !Partial)\n                unpack_low_range<SPLIT_POINT>(position, count, output + position, 0);"):]
tail_new = """            if (position < count) {
                const size_t tail = count - position;
                if constexpr (SPLIT_POINT > 0) unpack_low_range<SPLIT_POINT>(position, count, blk, 0);
                uint64_t carry = carried();
                for (size_t i = 0; i < tail; ++i) {
                    const size_t at = position + i;
                    if ((bits[at >> 6] >> (at & 63)) & 1) carry = run_value(runs++);
                    if constexpr (SPLIT_POINT == 0) blk[i] = static_cast<T>(carry);
                    else blk[i] = static_cast<T>(carry + static_cast<U>(blk[i]));
                }
                fp::scaled_to_double(blk, doutput + position, tail, scale, inv);
            }
        }

"""
fp = fp.replace(tail_old, tail_new)
api = """
        template<uint8_t SPLIT_POINT>
        size_t get_elements_fp_worker(size_t startIndex, size_t count, double* out, double scale, double inv) const {
            if (count == 0 || startIndex >= m_num_elements) [[unlikely]] return 0;
            const size_t actual = std::min(count, m_num_elements - startIndex);
            if constexpr (!ExceptionBitVectorType::reverse_bit_order) {
                if (h != 0 && startIndex == 0 && actual == m_num_elements && actual >= 64) {
                    reconstruct_marker_words_fp<SPLIT_POINT>(actual, out, scale, inv);
                    return actual;
                }
            }
            // Other ranges: decode integers in L1-sized chunks (16 KB) and convert them.
            alignas(64) T buffer[2048];
            for (size_t done = 0; done < actual; done += 2048) {
                const size_t len = std::min<size_t>(2048, actual - done);
                get_elements_worker<SPLIT_POINT>(startIndex + done, len, buffer);
                fp::scaled_to_double(buffer, out + done, len, scale, inv);
            }
            return actual;
        }

        template <size_t... Is>
        size_t dispatch_fp_worker(size_t b_val, size_t start, size_t count, double* out, double scale, double inv,
                                  std::index_sequence<Is...>) const {
            using WorkerPtr = size_t (RLE_GEF_FP::*)(size_t, size_t, double*, double, double) const;
            static constexpr WorkerPtr table[] = { &RLE_GEF_FP::get_elements_fp_worker<Is>... };
            if (b_val >= sizeof...(Is)) throw std::invalid_argument("Invalid b value");
            return (this->*table[b_val])(start, count, out, scale, inv);
        }

    public:
        // Decodes [startIndex, startIndex + count) as doubles v / scale (correctly rounded,
        // inv = 1 / scale), converting in registers before the only store of each value.
        size_t get_elements_into(size_t startIndex, size_t count, double* output, double scale, double inv) const {
            if (count == 0 || startIndex >= m_num_elements) return 0;
            const size_t actual = std::min(count, m_num_elements - startIndex);
            if constexpr (RandomAccess) {
                return dispatch_fp_worker(b, startIndex, actual, output, scale, inv,
                                          std::make_index_sequence<sizeof(T) * 8 + 1>{});
            } else {
                std::vector<T> tmp(actual);
                get_elements_into(startIndex, actual, tmp.data());
                fp::scaled_to_double(tmp.data(), output, actual, scale, inv);
                return actual;
            }
        }
    private:
"""
s = s[:end] + fp + api + s[end:]
write("RLE_GEF_FP.hpp", s)

# ---------------------------------------------------------------- B*-GEF fork
src = read("B_STAR_GEF.hpp")
s = re.sub(r"\bB_STAR_GEF\b", "B_STAR_GEF_FP", src)
s = s.replace("#ifndef B_STAR_GEF_HPP\n#define B_STAR_GEF_HPP", "#ifndef B_STAR_GEF_FP_HPP\n#define B_STAR_GEF_FP_HPP", 1)
# Namespace-scope helpers exist in the original header too: reuse those instead of redefining.
a = s.index("    struct BStarZeroByte {")
b = s.index("    template<typename T, typename GapBitVectorType", a)
s = s[:a] + s[b:]
s = s.replace("#include \"IGEF.hpp\"", "#include \"IGEF.hpp\"\n#include \"FpConvert.hpp\"\n#include \"B_STAR_GEF.hpp\"", 1)
start = s.index("        template<uint8_t SPLIT_POINT, typename Position>\n        size_t decode_position_ranges(")
end = s.index("        template<uint8_t SPLIT_POINT>\n        size_t get_elements_worker(", start)
k = s[start:end]
fp = k
def rep(a_, b_):
    global fp
    assert fp.count(a_) == 1, a_
    fp = fp.replace(a_, b_)
rep("size_t decode_position_ranges(size_t startIndex, size_t count, T* output) const {",
    "size_t decode_position_ranges_fp(size_t startIndex, size_t count, double* doutput, double scale, double inv) const {")
rep("            alignas(64) Position minus_positions[MAX_BATCH];",
    "            alignas(64) Position minus_positions[MAX_BATCH];\n            alignas(64) T batch_values[MAX_BATCH]; // L1: reconstructed, then converted")
rep("                    output[produced + i] = static_cast<T>(base_u + (high | low));",
    "                    batch_values[i] = static_cast<T>(base_u + (high | low));")
rep("                                                           base_u, output + produced + i,",
    "                                                           base_u, batch_values + i,")
rep("                for (; i < batch; ++i) reconstruct_scalar(i);\n                produced += batch;",
    "                for (; i < batch; ++i) reconstruct_scalar(i);\n                fp::scaled_to_double_prefetch(batch_values, doutput + produced, batch, scale, inv,\n                                              doutput + count);\n                produced += batch;")
api = """
        template<uint8_t SPLIT_POINT>
        size_t get_elements_fp_worker(size_t startIndex, size_t count, double* out, double scale, double inv) const {
            if (count == 0 || startIndex >= m_num_elements) [[unlikely]] return 0;
            count = std::min(count, m_num_elements - startIndex);
            if (h != 0) {
#if defined(__AVX2__) && !defined(GEF_DISABLE_SIMD) && SIZE_MAX == UINT64_MAX
                if constexpr (sizeof(T) == 8 && !GapBitVectorType::reverse_bit_order) {
                    if (G_plus->size() <= INT32_MAX && G_minus->size() <= INT32_MAX)
                        return decode_position_ranges_fp<SPLIT_POINT, uint32_t>(startIndex, count, out, scale, inv);
                }
#endif
                return decode_position_ranges_fp<SPLIT_POINT, size_t>(startIndex, count, out, scale, inv);
            }
            alignas(64) T buffer[2048];
            for (size_t done = 0; done < count; done += 2048) {
                const size_t len = std::min<size_t>(2048, count - done);
                get_elements_worker<SPLIT_POINT>(startIndex + done, len, buffer);
                fp::scaled_to_double(buffer, out + done, len, scale, inv);
            }
            return count;
        }

        template <size_t... Is>
        size_t dispatch_fp_worker(size_t b_val, size_t start, size_t count, double* out, double scale, double inv,
                                  std::index_sequence<Is...>) const {
            using WorkerPtr = size_t (B_STAR_GEF_FP::*)(size_t, size_t, double*, double, double) const;
            static constexpr WorkerPtr table[] = { &B_STAR_GEF_FP::get_elements_fp_worker<Is>... };
            if (b_val >= sizeof...(Is)) throw std::invalid_argument("Invalid b value");
            return (this->*table[b_val])(start, count, out, scale, inv);
        }

    public:
        // Decodes [startIndex, startIndex + count) as correctly rounded doubles v / scale.
        size_t get_elements_into(size_t startIndex, size_t count, double* output, double scale, double inv) const
            requires (RandomAccess) {
            return dispatch_fp_worker(b, startIndex, count, output, scale, inv, std::make_index_sequence<65>{});
        }
    private:
"""
s = s[:start] + k + fp + api + s[end:]
write("B_STAR_GEF_FP.hpp", s)
