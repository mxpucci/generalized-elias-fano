#pragma once

#include "B_GEF.hpp"
#include "B_STAR_GEF.hpp"
#include "RLE_GEF.hpp"
#include "U_GEF.hpp"
#include "UniformPartitioning.hpp"

namespace gef {
    
    template <typename T, size_t partition_size = 32000, bool random_access = true>
    class B_GEF : public UniformPartitioning<T, internal::B_GEF<T, PastaExceptionBitVector, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy> {
        using Base = UniformPartitioning<T, internal::B_GEF<T, PastaExceptionBitVector, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy>;
    public:
        explicit B_GEF(const std::vector<T>& data,
                       SplitPointStrategy strategy = OPTIMAL_SPLIT_POINT)
            : Base(data, strategy) {}

        B_GEF() = default;
    };

    template <typename T, size_t partition_size = 32000, bool random_access = true>
    class B_GEF_APPROXIMATE : public UniformPartitioning<T, internal::B_GEF<T, PastaExceptionBitVector, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy> {
        using Base = UniformPartitioning<T, internal::B_GEF<T, PastaExceptionBitVector, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy>;
    public:
        explicit B_GEF_APPROXIMATE(const std::vector<T>& data,
                                   SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT)
            : Base(data, strategy) {}

        B_GEF_APPROXIMATE() = default;
    };

    template <typename T, size_t partition_size = 32000, bool random_access = true>
    class B_STAR_GEF : public UniformPartitioning<T, internal::B_STAR_GEF<T, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy> {
        using Base = UniformPartitioning<T, internal::B_STAR_GEF<T, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy>;
    public:
        explicit B_STAR_GEF(const std::vector<T>& data,
                            SplitPointStrategy strategy = OPTIMAL_SPLIT_POINT)
            : Base(data, strategy) {}

        B_STAR_GEF() = default;
    };

    template <typename T, size_t partition_size = 32000, bool random_access = true>
    class RLE_GEF : public UniformPartitioning<T, internal::RLE_GEF<T, PastaRankBitVector, random_access>, partition_size> {
        using Base = UniformPartitioning<T, internal::RLE_GEF<T, PastaRankBitVector, random_access>, partition_size>;
    public:
        explicit RLE_GEF(const std::vector<T>& data)
            : Base(data) {}

        RLE_GEF() = default;
    };

    template <typename T, size_t partition_size = 32000, bool random_access = true>
    class U_GEF : public UniformPartitioning<T, internal::U_GEF<T, PastaExceptionBitVector, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy> {
        using Base = UniformPartitioning<T, internal::U_GEF<T, PastaExceptionBitVector, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy>;
    public:
        explicit U_GEF(const std::vector<T>& data,
                       SplitPointStrategy strategy = OPTIMAL_SPLIT_POINT)
            : Base(data, strategy) {}

        U_GEF() = default;
    };

    template <typename T, size_t partition_size = 32000, bool random_access = true>
    class U_GEF_APPROXIMATE : public UniformPartitioning<T, internal::U_GEF<T, PastaExceptionBitVector, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy> {
        using Base = UniformPartitioning<T, internal::U_GEF<T, PastaExceptionBitVector, PastaGapBitVector, random_access>, partition_size, SplitPointStrategy>;
    public:
        explicit U_GEF_APPROXIMATE(const std::vector<T>& data,
                                   SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT)
            : Base(data, strategy) {}

        U_GEF_APPROXIMATE() = default;
    };
}
