#ifndef GEF_UTILS_HPP
#define GEF_UTILS_HPP

#include <vector>
#include <string>
#include <limits>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cerrno>

template<typename TypeIn, typename TypeOut>
std::vector<TypeOut> read_data_binary(const std::string &filename, bool first_is_size = true,
                                      size_t max_size = std::numeric_limits<size_t>::max()) {
    try {
        auto openmode = std::ios::in | std::ios::binary;
        // If size isn't in the file, we need std::ios::ate to find the file size.
        if (!first_is_size) {
            openmode |= std::ios::ate;
        }

        std::ifstream in(filename, openmode);
        // Throw exceptions on I/O errors.
        in.exceptions(std::ios::failbit | std::ios::badbit);

        size_t total_elements_to_read;

        if (first_is_size) {
            // Read the total number of elements from the file's header.
            // Using a fixed-width type is good practice, but we'll stick to size_t
            // to match the original logic for 64-bit systems.
            in.read(reinterpret_cast<char*>(&total_elements_to_read), sizeof(size_t));
        } else {
            // Calculate size from the file's total size in bytes.
            std::streampos file_size_bytes = in.tellg();
            if (file_size_bytes < 0) { // Error check for tellg
                throw std::ios_base::failure("Could not determine file size.");
            }
            total_elements_to_read = static_cast<size_t>(file_size_bytes) / sizeof(TypeIn);
            // Seek back to the beginning to start reading the data.
            in.seekg(0);
        }

        // Apply the user-defined limit on the number of elements to read.
        total_elements_to_read = std::min(max_size, total_elements_to_read);

        // --- Robust Chunking Implementation ---

        std::vector<TypeIn> data;
        if (total_elements_to_read == 0) { // Handle case of empty or limited-to-zero file.
            return {};
        }

        // Pre-allocate the full memory needed to avoid costly reallocations.
        data.reserve(total_elements_to_read);

        // Define a reasonable chunk size (e.g., 1MB buffer for 8-byte doubles).
        const size_t chunk_elements = 131072;
        std::vector<TypeIn> buffer(chunk_elements);

        size_t elements_read = 0;
        while (elements_read < total_elements_to_read) {
            // Calculate how many elements to read in the current chunk.
            const size_t elements_in_this_chunk = std::min(chunk_elements, total_elements_to_read - elements_read);

            // Read a chunk of data into the temporary buffer.
            in.read(reinterpret_cast<char*>(buffer.data()), elements_in_this_chunk * sizeof(TypeIn));

            // Efficiently append the chunk from the buffer to the main data vector.
            data.insert(data.end(), buffer.begin(), buffer.begin() + elements_in_this_chunk);

            elements_read += elements_in_this_chunk;
        }

        // The final conversion logic is unchanged.
        if constexpr (std::is_same<TypeIn, TypeOut>::value) {
            return data;
        }

        return std::vector<TypeOut>(data.begin(), data.end());

    } catch (const std::ios_base::failure &e) {
        // Provide a more informative error message.
        std::cerr << "I/O Error: " << e.what() << "\n"
                  << "System Error: " << std::strerror(errno) << " (errno: " << errno << ")\n"
                  << "Failed while processing file: " << filename << std::endl;
        exit(1);
    }
}

#endif //GEF_UTILS_HPP