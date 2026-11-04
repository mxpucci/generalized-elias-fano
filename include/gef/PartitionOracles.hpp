#pragma once

#include "Partitioning.hpp"

#include <array>
#include <type_traits>

namespace gef {
namespace internal {

inline uint64_t partition_high(uint64_t value, unsigned split) {
    return split == 64 ? 0 : value >> split;
}

inline uint64_t partition_distance(uint64_t left, uint64_t right) {
    return left > right ? left - right : right - left;
}

template<class T>
class PartitionInput {
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool> && sizeof(T) <= 8);

protected:
    const std::vector<T>& values_;
    PartitionOptions options_;
    T base_ = 0;
    unsigned width_ = 0;

    PartitionInput(const std::vector<T>& values, PartitionOptions options)
        : values_(values), options_(options) {
        if (!options.grid || !options.block_overhead_bits) {
            throw std::invalid_argument("partition grid and block overhead must be positive");
        }
        if (!values.empty()) {
            const auto bounds = std::minmax_element(values.begin(), values.end());
            base_ = *bounds.first;
            width_ = std::bit_width(static_cast<uint64_t>(static_cast<__int128>(*bounds.second) - base_));
        }
    }

    uint64_t normalized(size_t index) const {
        return static_cast<uint64_t>(static_cast<__int128>(values_[index]) - base_);
    }

    size_t slot(size_t endpoint) const {
        if (endpoint > size() || (endpoint != size() && endpoint % grid())) {
            throw std::invalid_argument("cost endpoint is off oracle grid");
        }
        return endpoint / grid() + (endpoint % grid() != 0);
    }

    void validate_interval(size_t begin, size_t end) const {
        if (begin >= end || end > size()) throw std::out_of_range("invalid cost interval");
        slot(begin);
        slot(end);
    }

    size_t entries() const { return size() / grid() + (size() % grid() != 0) + 1; }

public:
    using value_type = T;
    size_t size() const { return values_.size(); }
    size_t grid() const { return options_.grid; }
    uint64_t minimum_block_cost() const { return options_.block_overhead_bits; }
};

template<class T, bool Exact>
class BStarPartitionOracle : public PartitionInput<T> {
    using Input = PartitionInput<T>;
    using Wide = PartitionWide;
    using Input::values_;
    using Input::base_;
    using Input::width_;
    using Input::normalized;
    using Input::slot;
    std::vector<std::vector<Wide>> variation_;

    Wide variation(size_t begin, size_t end, unsigned split) const {
        const uint64_t first = partition_high(normalized(begin), split);
        const uint64_t boundary = begin ? partition_distance(first, partition_high(normalized(begin - 1), split)) : 0;
        return variation_[split][slot(end)] - variation_[split][slot(begin)] - boundary + first;
    }

public:
    BStarPartitionOracle(const std::vector<T>& values, PartitionOptions options = {}) : Input(values, options) {
        const unsigned levels = Exact ? std::max(1u, width_) : 1;
        variation_.resize(levels, std::vector<Wide>(this->entries()));
        for (unsigned split = 0; split < levels; ++split) {
            Wide total = 0;
            uint64_t previous = 0;
            for (size_t i = 0; i < values.size(); ++i) {
                const uint64_t current = partition_high(normalized(i), split);
                if (i) total += partition_distance(current, previous);
                previous = current;
                if ((i + 1) % this->grid() == 0 || i + 1 == values.size()) {
                    variation_[split][slot(i + 1)] = total;
                }
            }
        }
    }

    PartitionCost<T> operator()(size_t begin, size_t end) const {
        this->validate_interval(begin, end);
        const size_t count = end - begin;
        Wide best = Wide(count) * width_;
        unsigned best_split = width_;
        auto offer = [&](unsigned split, Wide bits) {
            if (bits < best) {
                best = bits;
                best_split = split;
            }
        };
        if (width_) {
            const Wide total = variation(begin, end, 0);
            if constexpr (Exact) {
                const int estimate = total ? std::max(0, int(std::floor(std::log2(
                    std::log(2.0L) * static_cast<long double>(total) / count)))) : 0;
                for (int split = std::max(0, estimate - 1); split <= std::min(int(width_) - 1, estimate + 4); ++split) {
                    offer(split, Wide(count) * (split + 2) + variation(begin, end, split));
                }
            } else {
                const uint64_t mean = static_cast<uint64_t>(total / count);
                const unsigned split = std::min(width_ - 1, mean ? unsigned(std::bit_width(mean) - 1) : 0);
                const Wide quotient = (total >> split) + ((total & ((Wide(1) << split) - 1)) != 0);
                offer(split, Wide(count) * (split + 3) + quotient);
            }
        }
        return {partition_bits(best + this->minimum_block_cost()),
                {base_, static_cast<uint8_t>(width_), static_cast<uint8_t>(best_split)}};
    }
};

}

template<class T>
using ExactBStarOracle = internal::BStarPartitionOracle<T, true>;

template<class T>
using BoundBStarOracle = internal::BStarPartitionOracle<T, false>;

template<class T>
class RLEPartitionOracle : public internal::PartitionInput<T> {
    using Input = internal::PartitionInput<T>;
    using Wide = internal::PartitionWide;
    using Input::normalized;
    using Input::slot;
    std::vector<std::array<size_t, 65>> histogram_;

public:
    RLEPartitionOracle(const std::vector<T>& values, PartitionOptions options = {}) : Input(values, options) {
        histogram_.resize(this->entries());
        std::array<size_t, 65> counts{};
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) ++counts[std::bit_width(normalized(i) ^ normalized(i - 1))];
            if ((i + 1) % this->grid() == 0 || i + 1 == values.size()) histogram_[slot(i + 1)] = counts;
        }
    }

    PartitionCost<T> operator()(size_t begin, size_t end) const {
        this->validate_interval(begin, end);
        std::array<size_t, 65> counts;
        for (unsigned width = 0; width <= this->width_; ++width) {
            counts[width] = histogram_[slot(end)][width] - histogram_[slot(begin)][width];
        }
        if (begin) --counts[std::bit_width(normalized(begin) ^ normalized(begin - 1))];
        unsigned width = this->width_;
        while (width && !counts[width]) --width;
        const uint64_t anchor = width == 64 ? 0 : (normalized(begin) >> width) << width;
        const T base = static_cast<T>(static_cast<__int128>(this->base_) + anchor);
        const size_t count = end - begin;
        Wide best = Wide(count) * width;
        unsigned best_split = width;
        size_t runs = 1;
        for (unsigned split = width; split > 0;) {
            runs += counts[split--];
            const Wide bits = Wide(count) * (split + 1) + Wide(width - split) * runs;
            if (bits < best) {
                best = bits;
                best_split = split;
            }
        }
        return {internal::partition_bits(best + this->minimum_block_cost()),
                {base, static_cast<uint8_t>(width), static_cast<uint8_t>(best_split)}};
    }
};

}
