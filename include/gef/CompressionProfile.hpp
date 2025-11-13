#ifndef GEF_COMPRESSION_PROFILE_HPP
#define GEF_COMPRESSION_PROFILE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace gef {

struct CompressionBuildMetrics {
    double split_point_seconds = 0.0;
    double allocation_seconds = 0.0;
    double population_seconds = 0.0;

    std::size_t partitions = 0;
    std::size_t elements_processed = 0;
    std::size_t total_exceptions = 0;

    std::size_t sum_split_points = 0;
    std::uint8_t min_split_point = std::numeric_limits<std::uint8_t>::max();
    std::uint8_t max_split_point = 0;

    void record_partition(double split_secs,
                          double allocation_secs,
                          double population_secs,
                          std::size_t elements,
                          std::size_t exceptions,
                          std::uint8_t split_point) {
        partitions += 1;
        elements_processed += elements;
        total_exceptions += exceptions;
        split_point_seconds += split_secs;
        allocation_seconds += allocation_secs;
        population_seconds += population_secs;
        sum_split_points += split_point;
        if (split_point < min_split_point) {
            min_split_point = split_point;
        }
        if (split_point > max_split_point) {
            max_split_point = split_point;
        }
    }

    double average_split_point() const {
        if (partitions == 0) return 0.0;
        return static_cast<double>(sum_split_points) / static_cast<double>(partitions);
    }
};

} // namespace gef

#endif // GEF_COMPRESSION_PROFILE_HPP

