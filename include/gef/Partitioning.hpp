#pragma once

#include "BlockParameters.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gef {

struct PartitionOptions {
    double epsilon = 0.1;
    size_t grid = 256;
    uint64_t block_overhead_bits = 256;
};

template<class T>
struct PartitionCost {
    uint64_t bits;
    BlockParameters<T> parameters;
};

struct PartitionPlan {
    std::vector<size_t> ends;
    uint64_t estimated_bits = 0;
};

namespace internal {

using PartitionWide = unsigned __int128;

inline uint64_t partition_bits(PartitionWide bits) {
    if (bits >= UINT64_MAX) throw std::overflow_error("partition cost exceeds 64 bits");
    return static_cast<uint64_t>(bits);
}

inline std::vector<size_t> partition_cuts(size_t size, size_t grid) {
    if (!grid) throw std::invalid_argument("partition grid must be positive");
    std::vector<size_t> cuts{0};
    for (size_t cut = 0; cut < size;) {
        cut += std::min(grid, size - cut);
        cuts.push_back(cut);
    }
    return cuts;
}

inline std::vector<uint64_t> partition_thresholds(uint64_t minimum, uint64_t maximum,
                                                   double epsilon) {
    std::vector<uint64_t> thresholds;
    for (uint64_t threshold = minimum; threshold < maximum;) {
        thresholds.push_back(threshold);
        const long double next = std::floor(static_cast<long double>(threshold) * (1 + epsilon));
        if (next >= static_cast<long double>(maximum)) break;
        threshold = std::max(threshold + 1, static_cast<uint64_t>(next));
    }
    thresholds.push_back(maximum);
    return thresholds;
}

}

template<class Oracle>
PartitionPlan partition_sequence(const Oracle& oracle, PartitionOptions options = {}, bool exact = false) {
    const bool finite = (std::bit_cast<uint64_t>(options.epsilon) & UINT64_C(0x7ff0000000000000)) !=
                        UINT64_C(0x7ff0000000000000);
    if (!options.grid || options.grid != oracle.grid() || !finite || options.epsilon <= 0 ||
        !oracle.minimum_block_cost()) {
        throw std::invalid_argument("invalid partition grid, epsilon, or minimum block cost");
    }
    if (!oracle.size()) return {};
    const auto cuts = internal::partition_cuts(oracle.size(), options.grid);
    const size_t last = cuts.size() - 1;
    std::vector<uint64_t> costs(cuts.size(), UINT64_MAX);
    std::vector<size_t> previous(cuts.size());
    costs[0] = 0;
    const auto thresholds = exact ? std::vector<uint64_t>{} :
        internal::partition_thresholds(oracle.minimum_block_cost(), oracle(0, oracle.size()).bits,
                                       options.epsilon);
    std::vector<size_t> window_end(thresholds.size());
    for (size_t start = 0; start < last; ++start) {
        auto relax = [&](size_t end) {
            const uint64_t edge = oracle(cuts[start], cuts[end]).bits;
            if (edge >= UINT64_MAX - costs[start]) return;
            const uint64_t candidate = costs[start] + edge;
            if (candidate < costs[end]) {
                costs[end] = candidate;
                previous[end] = start;
            }
        };
        if (exact) {
            for (size_t end = start + 1; end <= last; ++end) relax(end);
            continue;
        }
        relax(start + 1);
        for (size_t bucket = 0; bucket < thresholds.size(); ++bucket) {
            size_t& end = window_end[bucket];
            end = std::max(end, start);
            while (end < last && oracle(cuts[start], cuts[end + 1]).bits <= thresholds[bucket]) ++end;
            if (end > start) relax(end);
            if (end == last) break;
        }
    }
    if (costs[last] == UINT64_MAX) throw std::overflow_error("partition cost exceeds 64 bits");
    PartitionPlan plan;
    plan.estimated_bits = costs[last];
    for (size_t end = last; end; end = previous[end]) plan.ends.push_back(cuts[end]);
    std::reverse(plan.ends.begin(), plan.ends.end());
    return plan;
}

}
