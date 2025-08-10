#ifndef IBITVECTOR_HPP
#define IBITVECTOR_HPP

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <memory>


class IBitVector {
public:
    IBitVector() = default;

    virtual ~IBitVector() = default;

    IBitVector(const IBitVector &other) = default;

    IBitVector(IBitVector &&other) = default;

    IBitVector &operator=(const IBitVector &other) = default;

    IBitVector &operator=(IBitVector &&other) = default;

    static double rank_overhead_per_bit() { return 0.1; }
    static double select1_overhead_per_bit() { return 0.1; }
    static double select0_overhead_per_bit() { return 0.1; }

    virtual bool operator[](size_t index) const = 0;

    virtual void set(size_t index, bool value) = 0;

    /**
     * @brief Sets a contiguous range of bits to a specific value.
     * @param start The starting index of the range (inclusive).
     * @param count The number of bits to set.
     * @param value The boolean value (0 or 1) to set the bits to.
     */
    virtual void set_range(size_t start, size_t count, bool value) = 0;

    /**
     * @brief Writes a block of bits from an integer into the vector.
     * @param start_index The starting position in the bitvector.
     * @param bits The integer containing the bits to write.
     * @param num_bits The number of (least significant) bits from 'bits' to write.
     */
    virtual void set_bits(size_t start_index, uint64_t bits, uint8_t num_bits) = 0;


    virtual size_t size() const = 0;

    /**
     * @brief Count number of 1-bits in range [0, pos)
     * @param pos End position (exclusive)
     * @return Number of 1-bits before position pos
     * @throws std::out_of_range if pos > size()
     */
    virtual size_t rank(size_t pos) const = 0;

    /**
     * @brief Count number of 1-bits in range [start, end)
     * @param start Start position (inclusive)
     * @param end End position (exclusive)
     * @return Number of 1-bits in the range
     * @throws std::out_of_range if start > end or end > size()
     */
    size_t rank(size_t start, size_t end) const {
        if (start > end) {
            throw std::out_of_range("start must be <= end");
        }
        return rank(end) - rank(start);
    }

    /**
     * @brief Count number of 0-bits in range [0, pos)
     * @param pos End position (exclusive)  
     * @return Number of 0-bits before position pos
     */
    size_t rank0(size_t pos) const {
        return pos - rank(pos);
    }

    /**
     * @brief Count number of 0-bits in range [start, end)
     * @param start Start position (inclusive)
     * @param end End position (exclusive)
     * @return Number of 0-bits in the range
     */
    size_t rank0(size_t start, size_t end) const {
        return (end - start) - rank(start, end);
    }

    /**
     * @brief Find position of the k-th 1-bit
     * @param k Which 1-bit to find (0-indexed)
     * @return Position of the k-th 1-bit
     * @throws std::out_of_range if k >= number of 1-bits
     */
    virtual size_t select(size_t k) const = 0;

    /**
     * @brief Find position of the k-th 0-bit
     * @param k Which 0-bit to find (0-indexed)
     * @return Position of the k-th 0-bit
     * @throws std::out_of_range if k >= number of 0-bits
     */
    virtual size_t select0(size_t k) const = 0;

    virtual size_t size_in_bytes() const = 0;

    virtual size_t size_in_megabytes() const = 0;


    virtual void serialize(std::ofstream &out) const = 0;

    void serialize(const std::filesystem::path &filepath) const {
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file");
        }
        this->serialize(ofs);
        ofs.close();
    }


    bool empty() const {
        return size() == 0;
    }

    virtual std::unique_ptr<IBitVector> clone() const = 0;

    virtual void enable_rank() = 0;

    virtual void enable_select1() = 0;

    virtual void enable_select0() = 0;
};

#endif
