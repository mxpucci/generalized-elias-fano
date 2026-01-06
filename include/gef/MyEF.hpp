//
// Created by Gemini on 25/07/25.
//

#ifndef MY_EF_HPP
#define MY_EF_HPP

#include "gap_computation_utils.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits>
#include <chrono>
#include <stdexcept>
#include "IGEF.hpp"
#include "CompressionProfile.hpp"
#include "FastBitWriter.hpp"
#include "FastUnaryDecoder.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/PastaBitVector.hpp"

#if __has_include(<experimental/simd>) && !defined(GEF_DISABLE_SIMD)
#include <experimental/simd>
namespace stdx = std::experimental;
#define GEF_EXPERIMENTAL_SIMD_ENABLED
#endif

namespace gef {
    template<typename T, typename BitVectorType = PastaBitVector>
    class MyEF : public IGEF<T> {
    private:
        static uint8_t bits_for_range(const T min_val, const T max_val) {
            using WI = __int128;
            using WU = unsigned __int128;
            const WI min_w = static_cast<WI>(min_val);
            const WI max_w = static_cast<WI>(max_val);
            const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);
            if (range <= 1) return 1;
            size_t bits = 0;
            WU x = range - 1; 
            while (x > 0) { ++bits; x >>= 1; }
            return static_cast<uint8_t>(std::min<size_t>(bits, sizeof(T) * 8));
        }

        // The High-part bit-vector. 
        // Stores unary representation of gaps between high parts.
        // Equivalent to storing 1 at index (high_val + i).
        std::unique_ptr<BitVectorType> H;

        // low parts
        sdsl::int_vector<> L;

        // The split point
        uint8_t b;
        
        // Total number of elements
        size_t m_num_elements;

        /**
         * The minimum of the encoded sequence.
         */
        T base;

        // Helper to determine bits required for the high part universe
        static uint8_t calculate_split_point(size_t n, T min_val, T max_val) {
            if (n == 0) return 0;
            using WU = unsigned __int128;
            // u = max + 1. Since we shift by base, effectively range is max-min+1
            WU u = (static_cast<WU>(max_val) - static_cast<WU>(min_val)) + 1;
            
            // Formula: max(0, floor(log(u/n)))
            double val = static_cast<double>(u) / static_cast<double>(n);
            if (val <= 1.0) return 0;
            return static_cast<uint8_t>(std::floor(std::log2(val)));
        }

    public:
        using IGEF<T>::serialize;
        using IGEF<T>::load;

        ~MyEF() override = default;

        // Default constructor
        MyEF() : b(0), m_num_elements(0), base(0) {
        }

        // Copy Constructor
        MyEF(const MyEF &other)
            : IGEF<T>(other),
              L(other.L),
              b(other.b),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            if (other.H) {
                H = std::make_unique<BitVectorType>(*other.H);
                H->enable_select1();
            } else {
                H = nullptr;
            }
        }

        // Friend swap
        friend void swap(MyEF &first, MyEF &second) noexcept {
            using std::swap;
            swap(first.H, second.H);
            swap(first.L, second.L);
            swap(first.b, second.b);
            swap(first.m_num_elements, second.m_num_elements);
            swap(first.base, second.base);
        }

        // Copy Assignment
        MyEF &operator=(const MyEF &other) {
            if (this != &other) {
                MyEF temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // Move Constructor
        MyEF(MyEF &&other) noexcept
            : IGEF<T>(std::move(other)),
              H(std::move(other.H)),
              L(std::move(other.L)),
              b(other.b),
              m_num_elements(other.m_num_elements),
              base(other.base) {
            other.m_num_elements = 0;
            other.base = T{};
        }

        // Move Assignment
        MyEF &operator=(MyEF &&other) noexcept {
            if (this != &other) {
                H = std::move(other.H);
                L = std::move(other.L);
                b = other.b;
                m_num_elements = other.m_num_elements;
                base = other.base;
            }
            return *this;
        }

        // Main Constructor
        template<typename C>
        MyEF(const C &S,
            SplitPointStrategy /*strategy*/ = OPTIMAL_SPLIT_POINT, // Ignored, logic is fixed
            CompressionBuildMetrics* metrics = nullptr) {
            
            using clock = std::chrono::steady_clock;
            std::chrono::time_point<clock> split_start;
            if (metrics) {
                split_start = clock::now();
            }

            const size_t N = S.size();
            m_num_elements = N;
            if (N == 0) {
                b = 0;
                base = T{};
                H = nullptr;
                if (metrics) {
                    double split_seconds = std::chrono::duration<double>(clock::now() - split_start).count();
                    metrics->record_partition(split_seconds, 0.0, 0.0, 0, 0, 0);
                }
                return;
            }

            auto [min_it, max_it] = std::minmax_element(S.begin(), S.end());
            base = *min_it;
            const T max_val = *max_it;

            // 1. Calculate Split Point
            b = calculate_split_point(N, base, max_val);
            const uint8_t total_bits = bits_for_range(base, max_val);
            // Safety clamp
            if (b > total_bits) b = total_bits;

            double split_seconds = 0.0;
            if (metrics) {
                split_seconds = std::chrono::duration<double>(clock::now() - split_start).count();
            }

            std::chrono::time_point<clock> allocation_start;
            if (metrics) {
                allocation_start = clock::now();
            }

            // 2. Allocation
            if (b > 0) {
                L = sdsl::int_vector<>(N, 0, b);
            } else {
                L = sdsl::int_vector<>(0); // SDSL requires 0-width vectors to be size 0
            }

            // Calculate H size: N 1s + (MaxHigh) 0s. 
            // MaxHigh = (max - min) >> b.
            // Size approx N + (U / 2^b).
            using U = std::make_unsigned_t<T>;
            const U universe_span = static_cast<U>(max_val) - static_cast<U>(base);
            const U max_high = (b >= sizeof(U)*8) ? 0 : (universe_span >> b);
            
            // We need to store N ones and max_high zeros. Total bits: N + max_high + padding
            // In unary encoding: 00..01 means gap of zeros followed by terminator.
            const size_t h_bits = static_cast<size_t>(N) + static_cast<size_t>(max_high) + 64; // +64 safety
            H = std::make_unique<BitVectorType>(h_bits);

            double allocation_seconds = 0.0;
            if (metrics) {
                allocation_seconds = std::chrono::duration<double>(clock::now() - allocation_start).count();
            }

            std::chrono::time_point<clock> population_start;
            if (metrics) {
                population_start = clock::now();
            }

            // 3. Population
            uint64_t* h_data = H->raw_data_ptr();
            FastBitWriter<BitVectorType::reverse_bit_order> h_writer(h_data);

            const U low_mask = (b > 0) ? ((U(1) << b) - 1) : 0;
            U last_high = 0;

            for (size_t i = 0; i < N; ++i) {
                const U val = static_cast<U>(S[i]) - static_cast<U>(base);
                
                // Store Low part
                if (b > 0) {
                    L[i] = static_cast<typename sdsl::int_vector<>::value_type>(val & low_mask);
                }

                // Store High part (unary encoding of gaps)
                U current_high = (b >= sizeof(U)*8) ? 0 : (val >> b);
                U gap = current_high - last_high;
                
                // Write 'gap' zeros
                if (gap > 0) {
                    // FastBitWriter set_zero is implicit if memory is calloc'd, 
                    // but we must advance position.
                    // However, standard FastBitWriter writes 1s or 0s.
                    // Here we treat 0 as increment, 1 as stop.
                    // Actually, let's use the writer explicitly.
                    // Note: FastBitWriter usually has set_ones_range / set_zero.
                    // We assume the memory is zeroed initially or handled by writer.
                    // Based on B_GEF usage, we set 1s explicitly.
                    // We need to skip 'gap' bits (leave them 0) and write a 1.
                    h_writer.skip_zeros(static_cast<size_t>(gap));
                }
                
                // Write terminator 1
                h_writer.write_one();
                
                last_high = current_high;
            }

            H->enable_select1();
            
            // Resize H to actual used bits to save space (optional, dependent on BV implementation)
            // But usually fixed size based on estimate is safer for static BV types.

            if (metrics) {
                double population_seconds = std::chrono::duration<double>(clock::now() - population_start).count();
                metrics->record_partition(split_seconds,
                                          allocation_seconds,
                                          population_seconds,
                                          N,
                                          0, // No exceptions in Pure EF
                                          b);
            }
        }

        size_t get_elements(size_t startIndex, size_t count, std::vector<T>& output) const override {
            if (count == 0 || startIndex >= m_num_elements) {
                return 0;
            }
            if (output.size() < count) {
                throw std::invalid_argument("output buffer is smaller than requested count");
            }

            const size_t endIndex = std::min(startIndex + count, m_num_elements);
            size_t write_index = 0;

            using Wide = std::conditional_t<(sizeof(T) < 4), uint32_t, std::make_unsigned_t<T>>;
            using Acc = std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>;
            
            // Initialize High part decoding
            // We need to find the state of the high part at startIndex.
            // High val = position_of_1(startIndex) - startIndex.
            // Note: select1(k) returns index of k-th 1. Argument is 1-based usually in SDSL/Pasta.
            // Let's assume 1-based rank/select.
            
            size_t h_pos_bit_index = H->select(startIndex + 1);
            Acc current_high = static_cast<Acc>(h_pos_bit_index - startIndex);

            // Create decoder starting *after* the (startIndex)-th 1.
            // The decoder will read the gaps for subsequent elements.
            // We need to decode (endIndex - startIndex) values.
            // The first value (at startIndex) is already resolved via select.
            
            // 1. Process first element manually to align
            {
                Wide low = (b == 0) ? 0 : static_cast<Wide>(L[startIndex]);
                Wide high_shifted = static_cast<Wide>(current_high) << b;
                output[write_index++] = static_cast<T>(static_cast<Acc>(base) + static_cast<Acc>(high_shifted | low));
            }

            // 2. Process remaining elements sequentially using FastUnaryDecoder
            if (endIndex > startIndex + 1) {
                const size_t remaining = endIndex - (startIndex + 1);
                const size_t start_bit_for_decoder = h_pos_bit_index + 1;
                const size_t h_size = H->size();
                
                FastUnaryDecoder<BitVectorType::reverse_bit_order> h_decoder(H->raw_data_ptr(), h_size, start_bit_for_decoder);

                constexpr size_t GAP_BATCH = 64;
                uint32_t gap_buffer[GAP_BATCH];
                size_t gap_size = 0, gap_index = 0;

                for (size_t i = startIndex + 1; i < endIndex; ++i) {
                    if (gap_index >= gap_size) [[unlikely]] {
                        gap_size = h_decoder.next_batch(gap_buffer, GAP_BATCH);
                        gap_index = 0;
                        if (gap_size == 0) [[unlikely]] {
                            // Fallback for single/tail items
                             gap_buffer[0] = static_cast<uint32_t>(h_decoder.next());
                             gap_size = 1;
                        }
                    }

                    // Accumulate gap to high part
                    current_high += gap_buffer[gap_index++];
                    
                    Wide low = (b == 0) ? 0 : static_cast<Wide>(L[i]);
                    Wide high_shifted = static_cast<Wide>(current_high) << b;
                    output[write_index++] = static_cast<T>(static_cast<Acc>(base) + static_cast<Acc>(high_shifted | low));
                }
            }

            return write_index;
        }

        T operator[](size_t index) const override {
            using Wide = std::conditional_t<(sizeof(T) < 4), uint32_t, std::make_unsigned_t<T>>;
            using Acc = std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>;

            // 1. Retrieve Low part
            Wide low = (b == 0) ? 0 : static_cast<Wide>(L[index]);

            // 2. Retrieve High part
            // Definition: The i-th element's high part is encoded by the position of the i-th 1.
            // H_val = position(i-th 1) - i.
            // Assuming select1 is 1-based (select1(1) is first one).
            size_t pos = H->select(index + 1);
            Acc high_val = static_cast<Acc>(pos - index);

            // 3. Combine
            Wide high_shifted = static_cast<Wide>(high_val) << b;
            return static_cast<T>(static_cast<Acc>(base) + static_cast<Acc>(high_shifted | low));
        }

        void serialize(std::ofstream &ofs) const override {
            if (!ofs.is_open()) {
                throw std::runtime_error("Could not open file for serialization");
            }
            ofs.write(reinterpret_cast<const char *>(&b), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&m_num_elements), sizeof(m_num_elements));
            ofs.write(reinterpret_cast<const char *>(&base), sizeof(T));
            
            if (b > 0) {
                L.serialize(ofs);
            }
            
            if (H) {
                H->serialize(ofs);
            }
        }

        void load(std::ifstream &ifs) override {
            ifs.read(reinterpret_cast<char *>(&b), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&m_num_elements), sizeof(m_num_elements));
            ifs.read(reinterpret_cast<char *>(&base), sizeof(T));
            
            if (b > 0) {
                L.load(ifs);
            } else {
                L = sdsl::int_vector<>(0);
            }
            
            // Standard EF always has H if there are elements, 
            // but we check m_num_elements to be safe or if H was saved conditionally
            if (m_num_elements > 0) {
                H = std::make_unique<BitVectorType>(BitVectorType::load(ifs));
                H->enable_select1();
            } else {
                H = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return m_num_elements;
        }

        [[nodiscard]] size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (H) {
                total_bytes += H->size_in_bytes();
            }
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sizeof(base);
            total_bytes += sizeof(b);
            total_bytes += sizeof(m_num_elements);
            return total_bytes;
        }

        [[nodiscard]] size_t size_in_bytes_without_supports() const override {
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };
            size_t total_bytes = 0;
            
            total_bytes += sdsl::size_in_bytes(L);
            if (H) {
                // Approximate raw payload of bitvector (implementation dependent, usually size/8)
                total_bytes += bits_to_bytes(H->size());
            }
            
            total_bytes += sizeof(base);
            total_bytes += sizeof(b);
            total_bytes += sizeof(m_num_elements);
            return total_bytes;
        }

        [[nodiscard]] size_t theoretical_size_in_bytes() const override {
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };
            size_t total_bytes = 0;
            
            // L vector: N * b
            total_bytes += bits_to_bytes(m_num_elements * b);
            
            // H vector: N + (U / 2^b) bits
            if (H) {
                total_bytes += bits_to_bytes(H->size());
            }
            
            total_bytes += sizeof(base);
            total_bytes += sizeof(b);
            total_bytes += sizeof(m_num_elements);
            
            return total_bytes;
        }

        [[nodiscard]] uint8_t split_point() const override {
            return this->b;
        }
    };
} // namespace gef

#endif