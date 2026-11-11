#pragma once
// Floating-point forks of RLE-GEF and B*-GEF: the same structures over scaled
// integers v = x * 10^p, with double input and double output. Decoding converts
// in registers before each value's only store; see FpConvert.hpp.
//
// Input values must have at most p decimal digits (x == RN(v / 10^p)).
#include <vector>
#include "gef.hpp"
#include "FpConvert.hpp"
#include "RLE_GEF_FP.hpp"
#include "B_STAR_GEF_FP.hpp"

namespace gef {
namespace fp {
// Per-partition choice of the conversion formula, set at compression. A separate
// base so that it is constructed before the partitions that fill it.
struct PartitionConversion {
    double m_scale = 1, m_inv = 1, m_lo = 0;
    std::vector<uint8_t> m_two_op; // 1: the two-operation formula reproduces the partition exactly
    PartitionConversion() = default;
    PartitionConversion(int64_t decimals, size_t n, size_t partition_size)
        : m_scale(scale_for(decimals)), m_inv(1.0 / scale_for(decimals)), m_lo(low_part(m_scale, m_inv)),
          m_two_op((n + partition_size - 1) / partition_size, 0) {}
    bool two_op(size_t partition) const { return partition < m_two_op.size() && m_two_op[partition]; }
    // Converts one partition's doubles and records whether the cheaper formula is exact on it.
    template <class T>
    void fill(const double* data, size_t start, size_t len, T* dst, size_t partition_size) {
        double_to_scaled(data + start, dst, len, m_scale);
        m_two_op[start / partition_size] = two_op_exact(dst, data + start, len, m_scale, m_inv);
    }
    size_t flag_bytes() const { return (m_two_op.size() + 7) / 8; } // one bit per partition
};
} // namespace fp

template <class Base, size_t partition_size>
class FpCodec : private fp::PartitionConversion, public Base {
public:
    FpCodec() = default;
    template <class... Args>
    FpCodec(const double* data, size_t n, int64_t decimals, Args... args)
        : fp::PartitionConversion(decimals, n, partition_size),
          Base(typename Base::from_blocks_t{}, n,
               [&](size_t start, size_t len, int64_t* dst) { this->fill(data, start, len, dst, partition_size); },
               args...) {}
    // Decodes [start, start + count) as doubles into out.
    size_t get_doubles(size_t start, size_t count, double* out) const {
        size_t written = 0;
        const bool stream = count >= GEF_FP_PREFETCH_MIN_VALUES;
        this->for_each_range(start, count, [&](const auto& c, size_t local, size_t len, size_t offset) {
            written += c.get_elements_into(local, len, out + offset, m_scale, m_inv, stream,
                                           this->two_op((start + offset) / partition_size));
        });
        return written;
    }
    double at_double(size_t i) const {
        return fp::one(static_cast<int64_t>((*this)[i]), m_scale, m_inv, m_lo, this->two_op(i / partition_size));
    }
    // The encoded partitions plus one bit per partition for the conversion choice.
    size_t size_in_bytes() const override { return Base::size_in_bytes() + flag_bytes(); }
};

template <typename T = int64_t, size_t partition_size = 32000, bool random_access = true>
class RLE_GEF_FP : public FpCodec<UniformPartitioning<T, internal::RLE_GEF_FP<T, PastaRankBitVector, random_access>,
                                                      partition_size>, partition_size> {
    using Codec = FpCodec<UniformPartitioning<T, internal::RLE_GEF_FP<T, PastaRankBitVector, random_access>,
                                              partition_size>, partition_size>;
public:
    RLE_GEF_FP() = default;
    // Compresses n doubles with p decimal digits.
    RLE_GEF_FP(const double* data, size_t n, int64_t decimals) : Codec(data, n, decimals) {}
};

template <typename T = int64_t, size_t partition_size = 32000, bool random_access = true>
class B_STAR_GEF_APPROXIMATE_FP
    : public FpCodec<UniformPartitioning<T, internal::B_STAR_GEF_FP<T, PastaGapBitVector, random_access>, partition_size,
                                         SplitPointStrategy>, partition_size> {
    using Codec = FpCodec<UniformPartitioning<T, internal::B_STAR_GEF_FP<T, PastaGapBitVector, random_access>,
                                              partition_size, SplitPointStrategy>, partition_size>;
public:
    B_STAR_GEF_APPROXIMATE_FP() = default;
    // Compresses n doubles with p decimal digits (approximate split point).
    B_STAR_GEF_APPROXIMATE_FP(const double* data, size_t n, int64_t decimals)
        : Codec(data, n, decimals, APPROXIMATE_SPLIT_POINT) {}
};

} // namespace gef
