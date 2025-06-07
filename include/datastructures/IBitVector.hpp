#ifndef IBITVECTOR_HPP
#define IBITVECTOR_HPP

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <filesystem>


class IBitVector {
public:
    virtual ~IBitVector() = default;

    virtual bool operator[](size_t index) const = 0;
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

    virtual void serialize(const std::filesystem::path& filepath) const = 0;
    
    
    bool empty() const {
        return size() == 0;
    }

    virtual void enable_rank() = 0;
    virtual void enable_select1() = 0;
    virtual void enable_select0() = 0;
};

#endif
