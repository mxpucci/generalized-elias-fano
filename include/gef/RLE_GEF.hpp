#ifndef RLE_GEF_H
#define RLE_GEF_H

#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits> // Required for std::make_unsigned
#include <utility>
#include <stdexcept>
#include "IGEF.hpp"
#include "FastBitWriter.hpp"
#include "NoRandomAccess.hpp"

#include "../datastructures/IBitVector.hpp"
#include "../datastructures/SDSLBitVector.hpp"
#include "../datastructures/PastaBitVector.hpp"

#if __has_include(<experimental/simd>) && !defined(GEF_DISABLE_SIMD)
#include <experimental/simd>
namespace stdx = std::experimental;
#define GEF_EXPERIMENTAL_SIMD_ENABLED
#endif

namespace gef {
    namespace internal {
    template<typename T, typename ExceptionBitVectorType = PastaRankBitVector, bool RandomAccess = true>
    class RLE_GEF : public IGEF<T> {
    public:
        // Bit-vector such that B[i] = 1 <==> highPart(i) != highPart(i - 1)
        std::unique_ptr<ExceptionBitVectorType> B;

        // high parts
        sdsl::int_vector<> H;

        // low parts
        sdsl::int_vector<> L;

        // The split point that rules which bits are stored in H and in L
        uint8_t b;
        uint8_t h;
        size_t m_num_elements;

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

        template<typename C>
        static uint8_t optimal_split_point(const C& S, const uint8_t total_bits, const T min) {
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
        RLE_GEF() : h(0), b(0), m_num_elements(0), base(0) {
        }

        // 2. Copy Constructor
        RLE_GEF(const RLE_GEF &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              H(other.H),
              L(other.L),
              h(other.h),
              b(other.b),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            if (other.h > 0) {
                B = std::make_unique<ExceptionBitVectorType>(*other.B);
                if constexpr (RandomAccess) {
                    B->enable_rank();
                }

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
            swap(first.m_num_elements, second.m_num_elements);
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
              m_num_elements(other.m_num_elements),
              base(other.base) {
            // Leave the moved-from object in a valid, empty state
            other.h = 0;
            other.m_num_elements = 0;
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
                m_num_elements = other.m_num_elements;
                base = other.base;
            }
            return *this;
        }


        // Constructor
        template<typename C>
        RLE_GEF(const C &S) {
            const size_t N = S.size();
            m_num_elements = N;
            if (S.size() == 0) {
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

            if (b > 0) {
                L = sdsl::int_vector<>(S.size(), 0, b);
            } else {
                L = sdsl::int_vector<>(0);
            }

            if (h == 0) {
                // All in L - use unsigned arithmetic for efficiency
                using U = std::make_unsigned_t<T>;
                
                if (b > 0 && L.size() != S.size()) {
                    L = sdsl::int_vector<>(S.size(), 0, b);
                }

                if (b > 0) {
                    for (size_t i = 0; i < S.size(); i++) {
                        L[i] = static_cast<typename sdsl::int_vector<>::value_type>(
                            static_cast<U>(S[i]) - static_cast<U>(base)
                        );
                    }
                }
                B = nullptr;
                H.resize(0);
                return;
            }

            // Pass 1: Count unique high parts and populate L & B
            B = std::make_unique<ExceptionBitVectorType>(S.size());
            T lastHighBits = 0;
            size_t h_count = 0;
            uint64_t* b_data = B->raw_data_ptr();
            FastBitWriter<ExceptionBitVectorType::reverse_bit_order> b_writer(b_data);
            
            // Precompute low mask once - b < total_bits is guaranteed by h > 0
            using U = std::make_unsigned_t<T>;
            const U low_mask = (b > 0) ? ((U(1) << b) - 1) : 0;
            
            for (size_t i = 0; i < S.size(); ++i) {
                const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
                // Direct shift - b < total_bits is guaranteed by early h==0 return
                const T highBits = static_cast<T>(element_u >> b);
                if (b > 0) {
                    L[i] = static_cast<typename sdsl::int_vector<>::value_type>(element_u & low_mask);
                }
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
            if constexpr (RandomAccess) {
                B->enable_rank();
            }


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

    private:
        template<uint8_t SPLIT_POINT>
        size_t get_elements_worker(size_t startIndex, size_t count, std::vector<T>& output) const {
            if (count == 0 || startIndex >= m_num_elements) [[unlikely]] {
                return 0;
            }
            if (output.size() < count) [[unlikely]] {
                throw std::invalid_argument("output buffer is smaller than requested count");
            }

            const size_t endIndex = std::min(startIndex + count, m_num_elements);
            size_t produced_count = 0;

            // L vector setup
            const uint64_t* l_raw_data = nullptr;
            uint64_t l_buffer_curr = 0;
            size_t l_word_idx = 0;
            uint32_t l_bit_offset = 0;

            if constexpr (SPLIT_POINT > 0) {
                l_raw_data = L.data();
                const size_t l_bit_global_pos = startIndex * SPLIT_POINT;
                l_word_idx = l_bit_global_pos >> 6;
                l_bit_offset = static_cast<uint32_t>(l_bit_global_pos & 63);
                l_buffer_curr = l_raw_data[l_word_idx];
            }

            constexpr uint64_t l_mask = (SPLIT_POINT == 64) ? ~0ULL : ((1ULL << SPLIT_POINT) - 1);

            // Fast path: h == 0
            if (h == 0) [[unlikely]] {
                T* out_ptr = output.data();
                size_t current_pos = startIndex;
                while (current_pos < endIndex) {
                     uint64_t l_val = 0;
                     if constexpr (SPLIT_POINT > 0) {
                        uint64_t word0 = l_buffer_curr >> l_bit_offset;
                        if (l_bit_offset + SPLIT_POINT > 64) {
                            l_word_idx++;
                            l_buffer_curr = l_raw_data[l_word_idx];
                            uint64_t word1 = l_buffer_curr << (64 - l_bit_offset);
                            l_val = (word0 | word1) & l_mask;
                        } else {
                            l_val = word0 & l_mask;
                            if (l_bit_offset + SPLIT_POINT == 64) {
                                l_word_idx++;
                                l_buffer_curr = l_raw_data[l_word_idx];
                            }
                        }
                        l_bit_offset = (l_bit_offset + SPLIT_POINT) & 63;
                     }
                     out_ptr[produced_count++] = base + static_cast<T>(l_val);
                     current_pos++;
                }
                return produced_count;
            }

            // Normal path: h > 0
            size_t current_rank_idx = 0;
            if (startIndex < m_num_elements) {
                current_rank_idx = B->rank_unchecked(startIndex + 1);
                if (current_rank_idx == 0) current_rank_idx = 1;
            }

            T current_high = H[current_rank_idx - 1];
            T current_base_plus_high;
             if constexpr (SPLIT_POINT < sizeof(T) * 8) {
                 current_base_plus_high = base + (current_high << SPLIT_POINT);
             } else {
                 current_base_plus_high = base;
             }

            const uint64_t* b_data = B->raw_data_ptr();
            const size_t b_size = B->size();
            size_t current_pos = startIndex;
            T* out_ptr = output.data();

            while (current_pos < endIndex) {
                size_t next_one_pos = b_size;
                size_t search_start = current_pos + 1;
                
                size_t b_word_idx = search_start / 64;
                size_t b_bit_offset = search_start % 64;
                size_t max_words = (b_size + 63) / 64;
                
                if (b_word_idx < max_words) {
                     uint64_t word = b_data[b_word_idx];
                     uint64_t b_mask = (~0ULL) << b_bit_offset;
                     uint64_t masked_word = word & b_mask;
                     
                     if (masked_word != 0) {
                         next_one_pos = b_word_idx * 64 + __builtin_ctzll(masked_word);
                     } else {
                         for (size_t w = b_word_idx + 1; w < max_words; ++w) {
                             if (b_data[w] != 0) {
                                 next_one_pos = w * 64 + __builtin_ctzll(b_data[w]);
                                 break;
                             }
                         }
                     }
                }

                size_t run_limit = std::min(endIndex, next_one_pos);
                size_t count_in_run = run_limit - current_pos;
                
                for(size_t i=0; i<count_in_run; ++i) {
                    uint64_t l_val = 0;
                    if constexpr (SPLIT_POINT > 0) {
                        uint64_t word0 = l_buffer_curr >> l_bit_offset;
                        if (l_bit_offset + SPLIT_POINT > 64) {
                            l_word_idx++;
                            l_buffer_curr = l_raw_data[l_word_idx];
                            uint64_t word1 = l_buffer_curr << (64 - l_bit_offset);
                            l_val = (word0 | word1) & l_mask;
                        } else {
                            l_val = word0 & l_mask;
                             if (l_bit_offset + SPLIT_POINT == 64) {
                                l_word_idx++;
                                l_buffer_curr = l_raw_data[l_word_idx];
                            }
                        }
                        l_bit_offset = (l_bit_offset + SPLIT_POINT) & 63;
                    }
                    out_ptr[produced_count++] = current_base_plus_high + static_cast<T>(l_val);
                }
                current_pos = run_limit;

                if (current_pos == next_one_pos && current_pos < m_num_elements) {
                    current_rank_idx++;
                    if (current_rank_idx <= H.size()) {
                        current_high = H[current_rank_idx - 1];
                         if constexpr (SPLIT_POINT < sizeof(T) * 8) {
                             current_base_plus_high = base + (current_high << SPLIT_POINT);
                         }
                    }
                }
            }
            return produced_count;
        }

        template <size_t... Is>
        size_t dispatch_worker(size_t b_val, size_t start, size_t count, std::vector<T>& out, std::index_sequence<Is...>) const {
             using WorkerPtr = size_t (RLE_GEF::*)(size_t, size_t, std::vector<T>&) const;
             static constexpr WorkerPtr table[] = { &RLE_GEF::get_elements_worker<Is>... };
             if (b_val >= sizeof...(Is)) throw std::invalid_argument("Invalid b value");
             return (this->*table[b_val])(start, count, out);
        }

    public:
    private:
        size_t decode_prefix_no_support(size_t count, std::vector<T>& output) const {
            if (count == 0 || m_num_elements == 0) return 0;
            const size_t endIndex = std::min(count, m_num_elements);
            // Caller guarantees output has at least 'count' slots.

            using U = std::make_unsigned_t<T>;
            using Acc = std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>;

            if (h == 0) [[unlikely]] {
                if (b == 0) [[unlikely]] {
                    for (size_t i = 0; i < endIndex; ++i) output[i] = base;
                    return endIndex;
                }
                for (size_t i = 0; i < endIndex; ++i) {
                    const U low = static_cast<U>(L[i]);
                    const Acc sum = static_cast<Acc>(base) + static_cast<Acc>(low);
                    output[i] = static_cast<T>(sum);
                }
                return endIndex;
            }

            const uint64_t* b_data = B->raw_data_ptr();
            auto bit_at = [b_data](size_t pos) -> bool {
                return (b_data[pos >> 6] >> (pos & 63)) & 1ULL;
            };

            size_t h_idx = 0;
            U current_high = 0;

            for (size_t i = 0; i < endIndex; ++i) {
                if (bit_at(i)) {
                    current_high = static_cast<U>(H[h_idx++]);
                }
                const U low = (b > 0) ? static_cast<U>(L[i]) : U(0);
                const U offset = (b < sizeof(U) * 8) ? (low | (current_high << b)) : low;
                const Acc sum = static_cast<Acc>(base) + static_cast<Acc>(offset);
                output[i] = static_cast<T>(sum);
            }
            return endIndex;
        }

    public:
        size_t get_elements(size_t startIndex, size_t count, std::vector<T>& output) const override {
            if constexpr (RandomAccess) {
                // 65 is the number of cases (0 to 64)
                return dispatch_worker(b, startIndex, count, output, std::make_index_sequence<65>{});
            } else {
                return ::gef::internal::no_support::get_elements_force_from_zero<T>(
                    startIndex, count, m_num_elements, output,
                    [this](size_t n, std::vector<T>& out) { return this->decode_prefix_no_support(n, out); });
            }
        }

        T operator[](size_t index) const override {
            if constexpr (!RandomAccess) {
                std::vector<T> scratch;
                return ::gef::internal::no_support::at_force_from_zero<T>(
                    index, scratch,
                    [this](size_t n, std::vector<T>& out) { return this->decode_prefix_no_support(n, out); });
            }
            // Case 1: No high bits are used (h=0).
            // All information is stored in the L vector. Reconstruction is trivial.
            if (h == 0) [[unlikely]] {
                if (b == 0) [[unlikely]] return base;
                return base + L[index];
            }

            using U = std::make_unsigned_t<T>;
            // Initiate L access early to hide memory latency
            const U low = (b > 0) ? static_cast<U>(L[index]) : U(0);

            // Check if this position is an exception
            // Optimization: Use raw bit access for B
            const uint64_t* b_data = B->raw_data_ptr();
            bool is_exception = (b_data[index >> 6] >> (index & 63)) & 1;

            if (is_exception) [[unlikely]] {
                const size_t exception_rank = B->rank_unchecked(index + 1);
                const U high_val = static_cast<U>(H[exception_rank - 1]);
                return base + static_cast<T>(low | (high_val << b));
            }

            const size_t rank = B->rank_unchecked(index + 1);
            const T high_shifted = H[rank - 1] << b;
            return base + (low | high_shifted);
        }

        void serialize(std::ofstream &ofs) const override {
            if (!ofs.is_open()) {
                throw std::runtime_error("Could not open file for serialization");
            }
            ofs.write(reinterpret_cast<const char *>(&h), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&b), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&m_num_elements), sizeof(m_num_elements));
            ofs.write(reinterpret_cast<const char *>(&base), sizeof(T));
            
            if (b > 0) {
                L.serialize(ofs);
            }
            
            H.serialize(ofs);
            if (h > 0)
                B->serialize(ofs);
        }

        void load(std::ifstream &ifs) override {
            ifs.read(reinterpret_cast<char *>(&h), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&b), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&m_num_elements), sizeof(m_num_elements));
            ifs.read(reinterpret_cast<char *>(&base), sizeof(T));
            
            if (b > 0) {
                L.load(ifs);
            } else {
                L = sdsl::int_vector<>(0);
            }
            
            H.load(ifs);
            if (h > 0) {
                B = std::make_unique<ExceptionBitVectorType>(ExceptionBitVectorType::load(ifs));
                if constexpr (RandomAccess) {
                    B->enable_rank();
                }

            } else {
                B = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return m_num_elements;
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
    } // namespace internal
} // namespace gef

#endif


