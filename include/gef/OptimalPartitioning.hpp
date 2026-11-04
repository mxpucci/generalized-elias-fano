#pragma once

#include <fstream>
#include "IGEF.hpp"
#include "PartitionOracles.hpp"

#include <optional>
#include <utility>

namespace gef {

template<class T, class Compressor>
class OptimalPartitioning : public IGEF<T> {
    static_assert(std::is_base_of_v<IGEF<T>, Compressor>);
    std::vector<size_t> ends_;
    std::vector<std::optional<Compressor>> blocks_;
    uint64_t estimated_bits_ = 0;

    size_t block_index(size_t index) const {
        return std::upper_bound(ends_.begin(), ends_.end(), index) - ends_.begin();
    }

    size_t directory_bytes() const {
        return sizeof(uint64_t) + sizeof(size_t) * (1 + ends_.size());
    }

public:
    OptimalPartitioning() = default;
    OptimalPartitioning(const OptimalPartitioning&) = delete;
    OptimalPartitioning& operator=(const OptimalPartitioning&) = delete;
    OptimalPartitioning(OptimalPartitioning&&) noexcept = default;
    OptimalPartitioning& operator=(OptimalPartitioning&&) noexcept = default;

    template<class Oracle>
    OptimalPartitioning(const std::vector<T>& data, const Oracle& oracle, PartitionOptions options = {}) {
        if (data.size() != oracle.size()) throw std::invalid_argument("oracle and input sizes differ");
        auto plan = partition_sequence(oracle, options);
        ends_ = std::move(plan.ends);
        estimated_bits_ = plan.estimated_bits;
        blocks_.resize(ends_.size());
        size_t begin = 0;
        for (size_t block = 0; block < ends_.size(); ++block) {
            const size_t end = ends_[block];
            blocks_[block].emplace(Span<const T>(data.data() + begin, end - begin),
                                   oracle(begin, end).parameters);
            begin = end;
        }
    }

    size_t size() const override { return ends_.empty() ? 0 : ends_.back(); }
    uint8_t split_point() const override { return 0; }
    const std::vector<size_t>& partition_ends() const { return ends_; }
    uint64_t estimated_bits() const { return estimated_bits_; }

    size_t size_in_bytes() const override {
        size_t bytes = directory_bytes();
        for (const auto& block : blocks_) bytes += block->size_in_bytes();
        return bytes;
    }

    size_t size_in_bytes_without_supports() const override {
        size_t bytes = directory_bytes();
        for (const auto& block : blocks_) bytes += block->size_in_bytes_without_supports();
        return bytes;
    }

    size_t theoretical_size_in_bytes() const override {
        size_t bytes = directory_bytes();
        for (const auto& block : blocks_) bytes += block->theoretical_size_in_bytes();
        return bytes;
    }

    T operator[](size_t index) const override {
        if (index >= size()) throw std::out_of_range("index out of range");
        const size_t block = block_index(index);
        return (*blocks_[block])[index - (block ? ends_[block - 1] : 0)];
    }

    size_t get_elements(size_t start, size_t count, std::vector<T>& output) const override {
        if (!count || start >= size()) return 0;
        if (output.size() < count) throw std::invalid_argument("output buffer is smaller than requested count");
        count = std::min(count, size() - start);
        size_t written = 0;
        size_t block = block_index(start);
        std::vector<T> scratch;
        while (written < count) {
            const size_t length = std::min(count - written, ends_[block] - start);
            const size_t offset = start - (block ? ends_[block - 1] : 0);
            if (written == 0) {
                blocks_[block]->get_elements(offset, length, output);
            } else {
                scratch.resize(length);
                blocks_[block]->get_elements(offset, length, scratch);
                std::copy(scratch.begin(), scratch.end(), output.begin() + written);
            }
            written += length;
            start += length;
            ++block;
        }
        return written;
    }

    using IGEF<T>::serialize;
    using IGEF<T>::load;

    void serialize(std::ofstream& stream) const override {
        const size_t count = blocks_.size();
        stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
        stream.write(reinterpret_cast<const char*>(&estimated_bits_), sizeof(estimated_bits_));
        for (size_t end : ends_) stream.write(reinterpret_cast<const char*>(&end), sizeof(end));
        for (const auto& block : blocks_) block->serialize(stream);
        if (!stream) throw std::runtime_error("failed to serialize partitioned GEF");
    }

    void load(std::ifstream& stream) override {
        OptimalPartitioning restored;
        size_t count = 0;
        stream.read(reinterpret_cast<char*>(&count), sizeof(count));
        stream.read(reinterpret_cast<char*>(&restored.estimated_bits_), sizeof(estimated_bits_));
        if (!stream) throw std::runtime_error("invalid partitioned GEF header");
        restored.ends_.resize(count);
        size_t begin = 0;
        for (size_t& end : restored.ends_) {
            stream.read(reinterpret_cast<char*>(&end), sizeof(end));
            if (!stream || end <= begin) throw std::runtime_error("invalid partitioned GEF endpoints");
            begin = end;
        }
        restored.blocks_.resize(count);
        struct RestoreExceptions {
            std::ifstream& stream;
            std::ios_base::iostate mask;
            ~RestoreExceptions() {
                try { stream.exceptions(mask); } catch (...) {}
            }
        } restore{stream, stream.exceptions()};
        stream.exceptions(restore.mask | std::ios_base::failbit | std::ios_base::badbit);
        begin = 0;
        for (size_t i = 0; i < count; ++i) {
            restored.blocks_[i].emplace();
            restored.blocks_[i]->load(stream);
            if (!stream || restored.blocks_[i]->size() != restored.ends_[i] - begin) {
                throw std::runtime_error("invalid partitioned GEF block");
            }
            begin = restored.ends_[i];
        }
        *this = std::move(restored);
    }
};

}
