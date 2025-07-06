//
// Created by Michelangelo Pucci on 07/06/25.
//

#ifndef IGEF_HPP
#define IGEF_HPP

#include <vector>
#include <filesystem>
#include <memory>
#include <span>
#include <type_traits>
#include <string>
#include <stdexcept>

// C++20 concepts support
#if __cplusplus >= 202002L
#include <concepts>
#endif

namespace gef {

#if __cplusplus >= 202002L
template<typename T>
concept IntegralType = std::integral<T>;
template<IntegralType T>
#else
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
#endif
class IGEF {
    static_assert(std::is_integral_v<T>, "IGEF template parameter T must be an integral type");
    
public:
	virtual ~IGEF() = default;
	
	IGEF(const IGEF&) = default;
	IGEF(IGEF&&) = default;
	IGEF& operator=(const IGEF&) = default;
	IGEF& operator=(IGEF&&) = default;

	virtual size_t size() const = 0;
	
	virtual bool empty() const { return size() == 0; }
	
	virtual size_t size_in_bytes() const = 0;
	
	double size_in_megabytes() const {
		return static_cast<double>(size_in_bytes()) / (1024.0 * 1024.0); 
	}

	/**
	 * @brief Access element at given index
	 * @param index The index of the element to access
	 * @return The element at the specified index
	 * @throws std::out_of_range if index is out of bounds
	 */
	virtual T operator[](size_t index) const = 0;
	
	/**
	 * @brief Safe element access with bounds checking
	 * @param index The index of the element to access
	 * @return The element at the specified index
	 * @throws std::out_of_range if index is out of bounds
	 */
	virtual T at(size_t index) const {
		if (index >= size())
			throw std::out_of_range("index out of range");
		return operator[](index);
	}
	
	/**
	 * @brief Get a range of elements
	 * @param startIndex Starting index (inclusive)
	 * @param count Number of elements to retrieve
	 * @return Vector containing the requested elements
	 * @throws std::out_of_range if range is invalid
	 */
	std::vector<T> get_elements(size_t startIndex, size_t count) const {
		std::vector<T> result;
		result.reserve(size());
		for (size_t i = 0; i < count; i++) {
			result.emplace_back(at(startIndex + i));
		}
		return result;
	}
	
	/**
	 * @brief Get a range of elements using modern C++20 span-like interface
	 * @param startIndex Starting index (inclusive) 
	 * @param endIndex Ending index (exclusive)
	 * @return Vector containing elements in range [startIndex, endIndex)
	 * @throws std::out_of_range if range is invalid
	 */
	std::vector<T> get_range(size_t startIndex, size_t endIndex) const {
		if (startIndex >= endIndex || endIndex > size()) {
			throw std::out_of_range("Invalid range");
		}
		return get_elements(startIndex, endIndex - startIndex);
	}

	// ===== Serialization =====
	
	/**
	 * @brief Serialize the data structure to a file
	 * @param filepath Path where to write the serialized data
	 * @throws std::filesystem::filesystem_error on I/O errors
	 */
	void serialize(const std::filesystem::path& filepath) const {
		std::ofstream ofs(filepath, std::ios::binary);
		if (!ofs.is_open()) {
			throw std::runtime_error("Failed to open file");
		}
		serialize(ofs);
		ofs.close();
	}

	virtual void serialize(std::ofstream& ofs) const = 0;

	virtual void load(std::ifstream& ifs) = 0;

	virtual void load(std::filesystem::path& filepath, std::shared_ptr<IBitVectorFactory> bit_vector_factory) {
		std::ifstream ifs(filepath, std::ios::binary);
		if (!ifs.is_open()) {
			throw std::runtime_error("Failed to open file");
		}
		deserialize(filepath, bit_vector_factory);
		ifs.close();
	}

protected:
	IGEF() = default;
};

} // namespace gef

#endif //IGEF_HPP
