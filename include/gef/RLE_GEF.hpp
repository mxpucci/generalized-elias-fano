#ifndef RLE_GEF_H
#define RLE_GEF_H

#include <iostream>
#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits> // Required for std::make_unsigned
#include "IGEF.hpp"
#include "FastBitWriter.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"


namespace gef {
    template<typename T>
    class RLE_GEF : public IGEF<T> {
    public:
        // Bit-vector such that B[i] = 1 <==> highPart(i) != highPart(i - 1)
        std::unique_ptr<IBitVector> B;

        // high parts
        sdsl::int_vector<> H;

        // low parts
        sdsl::int_vector<> L;

        // The split point that rules which bits are stored in H and in L
        uint8_t b;
        uint8_t h;

        /**
         * The minimum of the encoded sequence, so that we store the shifted sequence
         * that falls in the range [0, max S - base]
         * This tricks may boost compression and allows us to implicitly store negative numbers
         */
        T base;

        /**
         * The longest common prefix of two integers x and y represented on total_bits
         */
        static uint8_t LCP(const T x, const T y, const uint8_t total_bits) {
            if (x == y)
                return total_bits;

            // Ensure bitwise operations are performed on unsigned types
            using UnsignedT = std::make_unsigned_t<T>;
            const UnsignedT ux = static_cast<UnsignedT>(x);
            const UnsignedT uy = static_cast<UnsignedT>(y);

            const UnsignedT diff = ux ^ uy;
            int leading_zeros;

            if constexpr (sizeof(T) == 8) {
                // uint64_t
                leading_zeros = __builtin_clzll(diff);
            } else if constexpr (sizeof(T) == 4) {
                // uint32_t
                leading_zeros = __builtin_clz(diff);
            } else if constexpr (sizeof(T) == 2) {
                // uint16_t
                leading_zeros = __builtin_clz(diff) - (sizeof(unsigned int) * 8 - 16);
            } else if constexpr (sizeof(T) == 1) {
                // uint8_t
                leading_zeros = __builtin_clz(diff) - (sizeof(unsigned int) * 8 - 8);
            } else {
                // Fallback for types not explicitly handled, though this should ideally be unreachable
                // for standard integral types. This might indicate a need for a more generic
                // __builtin_clz equivalent or a different approach for arbitrary T.
                // For now, we'll assume it's a power of 2 size and calculate based on that.
                leading_zeros = (8 * sizeof(T) - total_bits);
            }

            return leading_zeros - (8 * sizeof(T) - total_bits);
        }

        static uint8_t optimal_split_point(const std::vector<T> S, const uint8_t total_bits, const T min) {
            std::vector<size_t> lcp_frequencies(total_bits + 1, 0);
            for (size_t i = 1; i < S.size(); i++) {
                const uint8_t lcp = LCP(S[i] - min, S[i - 1] - min, total_bits);
                ++lcp_frequencies[lcp];
            }

            size_t best_b = total_bits;
            size_t best_space = total_bits * S.size();
            size_t rank_h = 1; // Assuming B[0] = 0
            for (uint8_t h_val = 1; h_val <= total_bits; h_val++) {
                const uint8_t b = total_bits - h_val;
                rank_h += lcp_frequencies[h_val - 1];
                const size_t space = S.size() * (b + 1) + rank_h * h_val;
                if (space < best_space) {
                    best_space = space;
                    best_b = b;
                }
            }
            return best_b;
        }

        static T highPart(const T x, const uint8_t total_bits, const uint8_t highBits) {
            const uint8_t lowBits = total_bits - highBits;
            // Cast to unsigned to ensure logical right shift
            return static_cast<T>(static_cast<std::make_unsigned_t<T>>(x) >> lowBits);
        }

        static T lowPart(const T x, const uint8_t lowBits) {
            if (lowBits >= sizeof(T) * 8) {
                return x;
            }
            // Cast to unsigned to ensure predictable bitwise AND
            const std::make_unsigned_t<T> mask = (static_cast<std::make_unsigned_t<T>>(1) << lowBits) - 1;
            return static_cast<T>(static_cast<std::make_unsigned_t<T>>(x) & mask);
        }

    public:
        using IGEF<T>::serialize;
        using IGEF<T>::load;

        ~RLE_GEF() override = default;

        // Default constructor
        RLE_GEF() : h(0), b(0), base(0) {
        }

        // 2. Copy Constructor
        RLE_GEF(const RLE_GEF &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              H(other.H),
              L(other.L),
              h(other.h),
              b(other.b),
              base(other.base) {
            if (other.h > 0) {
                B = other.B->clone();
                B->enable_rank();
            } else {
                B = nullptr;
            }
        }

        // Friend swap function for copy-and-swap idiom
        friend void swap(RLE_GEF &first, RLE_GEF &second) noexcept {
            using std::swap;
            swap(first.B, second.B);
            swap(first.H, second.H);
            swap(first.L, second.L);
            swap(first.h, second.h);
            swap(first.b, second.b);
            swap(first.base, second.base);
        }

        // 3. Copy Assignment Operator (using copy-and-swap idiom)
        RLE_GEF &operator=(const RLE_GEF &other) {
            if (this != &other) {
                RLE_GEF temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // 4. Move Constructor
        RLE_GEF(RLE_GEF &&other) noexcept
            : IGEF<T>(std::move(other)),
              B(std::move(other.B)),
              H(std::move(other.H)),
              L(std::move(other.L)),
              h(other.h),
              b(other.b),
              base(other.base) {
            // Leave the moved-from object in a valid, empty state
            other.h = 0;
            other.base = T{};
        }


        // 5. Move Assignment Operator
        RLE_GEF &operator=(RLE_GEF &&other) noexcept {
            if (this != &other) {
                B = std::move(other.B);
                H = std::move(other.H);
                L = std::move(other.L);
                h = other.h;
                b = other.b;
                base = other.base;
            }
            return *this;
        }


        // Constructor
        RLE_GEF(std::shared_ptr<IBitVectorFactory> bit_vector_factory,
                const std::vector<T> &S) {
            if (S.empty()) {
                b = 0;
                h = 0;
                base = T{};
                B = nullptr;
                return;
            }

            auto [min_it, max_it] = std::minmax_element(S.begin(), S.end());
            base = *min_it;
            const int64_t max_val = *max_it;
            const int64_t min_val = base;
            const uint64_t u = max_val - min_val + 1;
            const uint8_t total_bits = (u > 1) ? static_cast<uint8_t>(floor(log2(u)) + 1) : 1;


            b = optimal_split_point(S,
                                    total_bits,
                                    /* min= */ base);
            h = total_bits - b;

            L = sdsl::int_vector<>(S.size(), 0, b);
            if (h == 0) {
                // All in L - use unsigned arithmetic for efficiency
                using U = std::make_unsigned_t<T>;
                for (size_t i = 0; i < S.size(); i++) {
                    L[i] = static_cast<typename sdsl::int_vector<>::value_type>(
                        static_cast<U>(S[i]) - static_cast<U>(base)
                    );
                }
                B = nullptr;
                H.resize(0);
                return;
            }

            // Pass 1: Count unique high parts and populate L & B
            B = bit_vector_factory->create(S.size());
            T lastHighBits = 0;
            size_t h_count = 0;
            uint64_t* b_data = B->raw_data_ptr();
            FastBitWriter b_writer(b_data);
            
            // Precompute low mask once - b < total_bits is guaranteed by h > 0
            using U = std::make_unsigned_t<T>;
            const U low_mask = ((U(1) << b) - 1);
            
            for (size_t i = 0; i < S.size(); ++i) {
                const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
                // Direct shift - b < total_bits is guaranteed by early h==0 return
                const T highBits = static_cast<T>(element_u >> b);
                L[i] = static_cast<typename sdsl::int_vector<>::value_type>(element_u & low_mask);
                const bool is_new_run = (i == 0) | (highBits != lastHighBits);
                if (is_new_run) {
                    b_writer.set_ones_range(1);
                } else {
                    b_writer.set_zero();
                }
                h_count += is_new_run;
                lastHighBits = highBits;
            }
            assert(b_writer.position() == S.size());
            B->enable_rank();

            // Pass 2: Allocate exact size and populate H
            H = sdsl::int_vector<>(h_count, 0, h);
            lastHighBits = 0;
            size_t h_idx = 0;
            for (size_t i = 0; i < S.size(); ++i) {
                const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
                // Direct shift - b < total_bits is guaranteed by early h==0 return
                const T highBits = static_cast<T>(element_u >> b);
                if ((i == 0) | (highBits != lastHighBits)) {
                    H[h_idx++] = highBits;
                }
                lastHighBits = highBits;
            }
        }

        std::vector<T> get_elements(size_t startIndex, size_t count) const override {
            std::vector<T> result;
            result.reserve(count);
            
            if (count == 0 || startIndex >= size()) {
                return result;
            }
            
            const size_t endIndex = std::min(startIndex + count, size());
            
            // Fast path: h == 0, all data in L
            if (h == 0) {
                for (size_t i = startIndex; i < endIndex; ++i) {
                    result.push_back(base + L[i]);
                }
                return result;
            }
            
            // Optimized range access: reuse rank computations within runs
            size_t current_rank = B->rank(startIndex + 1);
            T current_high = H[current_rank - 1];
            
            for (size_t i = startIndex; i < endIndex; ++i) {
                // Check if we've crossed into a new run
                if ((*B)[i]) {
                    // New run started at position i
                    current_rank = B->rank(i + 1);
                    current_high = H[current_rank - 1];
                }
                result.push_back(base + (L[i] | (current_high << b)));
            }
            
            return result;
        }

        T operator[](size_t index) const override {
            if (h == 0)
                return base + L[index];

            const size_t rank = B->rank(index + 1);
            return base + (L[index] | (H[rank - 1] << b));
        }

        void serialize(std::ofstream &ofs) const override {
            if (!ofs.is_open()) {
                throw std::runtime_error("Could not open file for serialization");
            }
            ofs.write(reinterpret_cast<const char *>(&h), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&b), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&base), sizeof(T));
            L.serialize(ofs);
            H.serialize(ofs);
            if (h > 0)
                B->serialize(ofs);
        }

        void load(std::ifstream &ifs, const std::shared_ptr<IBitVectorFactory> bit_vector_factory) override {
            ifs.read(reinterpret_cast<char *>(&h), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&b), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&base), sizeof(T));
            L.load(ifs);
            H.load(ifs);
            if (h > 0) {
                B = bit_vector_factory->from_stream(ifs);
                B->enable_rank();
            } else {
                B = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return L.size();
        }

        [[nodiscard]] size_t size_in_bytes_without_supports() const override {
            size_t total_bytes = 0;
            if (B) {
                total_bytes += B->size_in_bytes() - B->support_size_in_bytes();
            }
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sdsl::size_in_bytes(H);
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            return total_bytes;
        }

        [[nodiscard]] size_t theoretical_size_in_bytes() const override {
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };
            size_t total_bytes = 0;
            
            // L vector: use width * size formula (theoretical)
            total_bytes += bits_to_bytes(L.size() * L.width());
            
            // H vector: use width * size formula (theoretical)
            total_bytes += bits_to_bytes(H.size() * H.width());
            
            // B bit vector (if it exists)
            if (B) {
                total_bytes += bits_to_bytes(B->size());
            }
            
            // Fixed metadata
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            
            return total_bytes;
        }

        [[nodiscard]] size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (B) {
                total_bytes += B->size_in_bytes();
            }
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sdsl::size_in_bytes(H);
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            return total_bytes;
        }

        [[nodiscard]] uint8_t split_point() const override {
            return this -> b;
        }


    };
} // namespace gef

#endif
