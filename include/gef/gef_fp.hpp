#pragma once
// Floating-point forks of RLE-GEF and B*-GEF: the same structures over scaled
// integers v = x * 10^p, with double input and double output. Decoding converts
// in registers before each value's only store; see FpConvert.hpp.
#include "gef.hpp"
#include "FpConvert.hpp"
#include "RLE_GEF_FP.hpp"
#include "B_STAR_GEF_FP.hpp"

namespace gef {

template <typename T = int64_t, size_t partition_size = 32000, bool random_access = true>
class RLE_GEF_FP : public UniformPartitioning<T, internal::RLE_GEF_FP<T, PastaRankBitVector, random_access>, partition_size> {
    using Base = UniformPartitioning<T, internal::RLE_GEF_FP<T, PastaRankBitVector, random_access>, partition_size>;
    double m_scale = 1, m_inv = 1;
public:
    RLE_GEF_FP() = default;
    // Compresses n doubles with p decimals; each partition is converted into a reused buffer.
    RLE_GEF_FP(const double* data, size_t n, int64_t decimals)
        : Base(typename Base::from_blocks_t{}, n,
               [&](size_t start, size_t len, T* dst) { fp::double_to_scaled(data + start, dst, len, fp::scale_for(decimals)); }),
          m_scale(fp::scale_for(decimals)), m_inv(1.0 / fp::scale_for(decimals)) {}
    size_t get_doubles(size_t start, size_t count, double* out) const {
        size_t written = 0;
        this->for_each_range(start, count, [&](const auto& c, size_t local, size_t len, size_t offset) {
            written += c.get_elements_into(local, len, out + offset, m_scale, m_inv);
        });
        return written;
    }
    double at_double(size_t i) const { return fp::scaled_to_double_one(static_cast<int64_t>((*this)[i]), m_scale, m_inv); }
};

template <typename T = int64_t, size_t partition_size = 32000, bool random_access = true>
class B_STAR_GEF_APPROXIMATE_FP : public UniformPartitioning<T, internal::B_STAR_GEF_FP<T, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy> {
    using Base = UniformPartitioning<T, internal::B_STAR_GEF_FP<T, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy>;
    double m_scale = 1, m_inv = 1;
public:
    B_STAR_GEF_APPROXIMATE_FP() = default;
    B_STAR_GEF_APPROXIMATE_FP(const double* data, size_t n, int64_t decimals)
        : Base(typename Base::from_blocks_t{}, n,
               [&](size_t start, size_t len, T* dst) { fp::double_to_scaled(data + start, dst, len, fp::scale_for(decimals)); },
               APPROXIMATE_SPLIT_POINT),
          m_scale(fp::scale_for(decimals)), m_inv(1.0 / fp::scale_for(decimals)) {}
    size_t get_doubles(size_t start, size_t count, double* out) const {
        size_t written = 0;
        this->for_each_range(start, count, [&](const auto& c, size_t local, size_t len, size_t offset) {
            written += c.get_elements_into(local, len, out + offset, m_scale, m_inv);
        });
        return written;
    }
    double at_double(size_t i) const { return fp::scaled_to_double_one(static_cast<int64_t>((*this)[i]), m_scale, m_inv); }
};

} // namespace gef
