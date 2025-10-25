//
// Created by Michelangelo Pucci on 03/08/25.
//

#ifndef B_GEF_NO_RLE_NO_RLE_HPP
#define B_GEF_NO_RLE_NO_RLE_HPP

#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits>
#include "IGEF.hpp"
#include "RLE_GEF.hpp"
#include "gap_computation_utils.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"


namespace gef {
    template<typename T>
    class B_GEF_STAR : public IGEF<T> {
    private:
        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that highPart(i) >= highPart(i - 1)
         */
        std::unique_ptr<IBitVector> G_plus;

        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that highPart(i - 1) >= highPart(i)
         */
        std::unique_ptr<IBitVector> G_minus;

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

        static size_t evaluate_space(const GapComputation &gap_computation, const uint8_t b) {
            // B_GEF_STAR allocates:
            // - L vector: N elements, b bits each (sdsl::int_vector)
            // - G_plus and G_minus: gap-encoded values
            // Conservative estimate without complex SDSL internal modeling
            const size_t N = gap_computation.negative_gaps + gap_computation.positive_gaps;
            if (N == 0) {
                return 256;
            }
            
            // Each component's storage need
            const size_t L_bits = N * static_cast<size_t>(b);
            const size_t G_plus_bits = gap_computation.sum_of_positive_gaps + N;
            const size_t G_minus_bits = gap_computation.sum_of_negative_gaps + N;
            
            // Sum with overflow protection
            size_t total_bits = 0;
            if (L_bits > SIZE_MAX - G_plus_bits - G_minus_bits) {
                return SIZE_MAX / 2;  // Avoid overflow
            }
            total_bits = L_bits + G_plus_bits + G_minus_bits;
            
            // Conservative estimate: account for SDSL overhead (align to bytes, add metadata)
            // Formula: roughly (total_bits / 8) * 1.5 for alignment + 256 for headers
            const size_t bytes_min = total_bits / 8;
            const size_t estimated = bytes_min + bytes_min / 2 + 256;  // 1.5x + overhead
            
            return estimated;
        }

        static double approximated_optimal_split_point(const std::vector<T> &S, const T min, const T max) {
            const GapComputation gap_computation = variation_of_original_vec(S, min, max);
            const size_t total_variation = gap_computation.sum_of_negative_gaps + gap_computation.sum_of_positive_gaps;

            return log2(log(2) * total_variation / S.size());
        }

        static std::pair<uint8_t, GapComputation> approximate_optimal_split_point(const std::vector<T> &S,
            const T min, const T max) {
            const double approx_b = approximated_optimal_split_point(S, min, max);
            
            if (ceil(approx_b) == floor(approx_b)) {
                const uint8_t b_clamped = std::max(0.0, std::min(64.0, floor(approx_b)));
                return {
                    b_clamped,
                    variation_of_shifted_vec(S, min, max, b_clamped, ExceptionRule::None)
                };
            }

            // Clamp to [0, 64] before casting to uint8_t to avoid underflow
            const uint8_t ceilB = std::max(0.0, std::min(64.0, ceil(approx_b)));
            const uint8_t floorB = std::max(0.0, std::min(64.0, floor(approx_b)));
            const std::vector<GapComputation> gap_computation =
                    total_variation_of_shifted_vec_with_multiple_shifts(S, min, max, floorB, ceilB, ExceptionRule::None);
            
            size_t best_index = 0;
            size_t best_space = SIZE_MAX;
            for (size_t i = 0; i < gap_computation.size(); i++) {
                const size_t space = evaluate_space(gap_computation[i], floorB + i);
                if (space < best_space) {
                    best_space = space;
                    best_index = i;
                }
            }

            const size_t total_bits = ceil(log2((max - min + 1)));
            // best_space is in bytes, so convert S.size() * total_bits from bits to bytes
            const size_t naive_space_bytes = ((S.size() * total_bits) + 7) / 8 + 64;  // +64 for metadata
            if (best_space > naive_space_bytes)
                return {total_bits, variation_of_shifted_vec(S, min, max, total_bits, ExceptionRule::None)};
            return {floorB + best_index, gap_computation[best_index]};
        }


        static std::pair<uint8_t, GapComputation>
        optimal_split_point(const std::vector<T> &S, const T min, const T max) {
            // B_GEF_STAR's optimal formula is inaccurate, so we use approximate for consistency
            return approximate_optimal_split_point(S, min, max);
        }

        static T highPart(const T x, const uint8_t total_bits, const uint8_t highBits) {
            const uint8_t lowBits = total_bits - highBits;
            return static_cast<T>(static_cast<std::make_unsigned_t<T>>(x) >> lowBits);
        }

        static T lowPart(const T x, const uint8_t lowBits) {
            if (lowBits >= sizeof(T) * 8) {
                return x;
            }
            const std::make_unsigned_t<T> mask = (static_cast<std::make_unsigned_t<T>>(1) << lowBits) - 1;
            return static_cast<T>(static_cast<std::make_unsigned_t<T>>(x) & mask);
        }

    public:
        using IGEF<T>::serialize;
        using IGEF<T>::load;

        ~B_GEF_STAR() override = default;

        // Default constructor
        B_GEF_STAR() : h(0), b(0), base(0) {
        }

        // 2. Copy Constructor
        B_GEF_STAR(const B_GEF_STAR &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              L(other.L),
              h(other.h),
              b(other.b),
              base(other.base) {
            if (other.h > 0) {
                G_plus = other.G_plus->clone();
                G_plus->enable_rank();
                G_plus->enable_select0();

                G_minus = other.G_minus->clone();
                G_minus->enable_rank();
                G_minus->enable_select0();
            } else {
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        // Friend swap function for copy-and-swap idiom
        friend void swap(B_GEF_STAR &first, B_GEF_STAR &second) noexcept {
            using std::swap;
            swap(first.L, second.L);
            swap(first.h, second.h);
            swap(first.b, second.b);
            swap(first.base, second.base);
            swap(first.G_plus, second.G_plus);
            swap(first.G_minus, second.G_minus);
        }

        // 3. Copy Assignment Operator (using copy-and-swap idiom)
        B_GEF_STAR &operator=(const B_GEF_STAR &other) {
            if (this != &other) {
                B_GEF_STAR temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // 4. Move Constructor
        B_GEF_STAR(B_GEF_STAR &&other) noexcept
            : IGEF<T>(std::move(other)),
              G_plus(std::move(other.G_plus)),
              G_minus(std::move(other.G_minus)),
              L(std::move(other.L)),
              h(other.h),
              b(other.b),
              base(other.base) {
            // Leave the moved-from object in a valid, empty state
            other.h = 0;
            other.base = T{};
        }


        // 5. Move Assignment Operator
        B_GEF_STAR &operator=(B_GEF_STAR &&other) noexcept {
            if (this != &other) {
                G_plus = std::move(other.G_plus);
                G_minus = std::move(other.G_minus);
                L = std::move(other.L);
                h = other.h;
                b = other.b;
                base = other.base;
            }
            return *this;
        }


        // Constructor
        B_GEF_STAR(const std::shared_ptr<IBitVectorFactory> &bit_vector_factory,
                   const std::vector<T> &S,
                   SplitPointStrategy strategy = OPTIMAL_SPLIT_POINT) {
            const size_t N = S.size();
            if (N == 0) {
                b = 0;
                h = 0;
                base = T{};
                return;
            }

            auto [min_it, max_it] = std::minmax_element(S.begin(), S.end());
            base = *min_it;
            const T max_val = *max_it;
            
            // Use 128-bit arithmetic to avoid overflow
            using WI = __int128;
            using WU = unsigned __int128;
            const WI min_w = static_cast<WI>(base);
            const WI max_w = static_cast<WI>(max_val);
            const WU range = static_cast<WU>(max_w - min_w) + static_cast<WU>(1);
            
            // Calculate total_bits safely, clamped to sizeof(T)*8
            uint8_t total_bits;
            if (range <= 1) {
                total_bits = 1;
            } else {
                size_t bits = 0;
                WU x = range - 1;
                while (x > 0) { ++bits; x >>= 1; }
                total_bits = std::min<size_t>(bits, sizeof(T) * 8);
            }
            
            GapComputation gap_computation;

            switch (strategy) {
                case APPROXIMATE_SPLIT_POINT:
                case OPTIMAL_SPLIT_POINT:
                    // B_GEF_STAR's optimal formula doesn't match SDSL behavior precisely,
                    // so we always use approximate strategy for correctness
                    std::tie(b, gap_computation) = approximate_optimal_split_point(S, base, max_val);
                    break;
            }
            h = total_bits - b;

            L = sdsl::int_vector<>(N, 0, b);
            if (h == 0) {
                // Special case: no high bits, only L is needed.
                for (size_t i = 0; i < N; ++i) {
                    L[i] = S[i] - base;
                }
                G_plus = nullptr;
                G_minus = nullptr;
                return;
            }

            const size_t g_plus_bits = gap_computation.sum_of_positive_gaps + N;
            const size_t g_minus_bits = gap_computation.sum_of_negative_gaps + N;

            G_plus = bit_vector_factory->create(g_plus_bits);
            G_minus = bit_vector_factory->create(g_minus_bits);

            size_t g_plus_pos = 0;
            size_t g_minus_pos = 0;
            using U = std::make_unsigned_t<T>;
            U lastHighBits = 0;

            // Unified loop - b==0 is rare, so don't split code paths
            const U low_mask = b ? (U(~U(0)) >> (sizeof(T) * 8 - b)) : U(0);

            for (size_t i = 0; i < N; ++i) {
                const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
                if (b > 0) [[likely]] {
                    L[i] = static_cast<typename sdsl::int_vector<>::value_type>(element_u & low_mask);
                }

                const U currentHighBits = element_u >> b;
                const int64_t gap = static_cast<int64_t>(currentHighBits) - static_cast<int64_t>(lastHighBits);

                // Branchless: use boolean to select which vector to write to
                const bool is_positive = gap > 0;
                const uint64_t abs_gap = is_positive ? static_cast<uint64_t>(gap) : static_cast<uint64_t>(-gap);
                
                if (is_positive) {
                    G_plus->set_range(g_plus_pos, abs_gap, true);
                    g_plus_pos += abs_gap;
                } else {
                    G_minus->set_range(g_minus_pos, abs_gap, true);
                    g_minus_pos += abs_gap;
                }
                
                // Always add terminators
                G_minus->set(g_minus_pos++, false);
                G_plus->set(g_plus_pos++, false);

                lastHighBits = currentHighBits;
            }

            // Enable rank/select support
            G_plus->enable_rank();
            G_plus->enable_select0();
            G_minus->enable_rank();
            G_minus->enable_select0();
        }

        T operator[](size_t index) const override {
            if (h == 0)
                return base + L[index];

            const T base_high_val = 0;
            const size_t pos_gap = G_plus->rank(G_plus->select0(index + 1));
            const size_t neg_gap = G_minus->rank(G_minus->select0(index + 1));
            const T high_val = base_high_val + pos_gap - neg_gap;

            return base + (L[index] | (high_val << b));
        }

        void serialize(std::ofstream &ofs) const override {
            if (!ofs.is_open()) {
                throw std::runtime_error("Could not open file for serialization");
            }
            ofs.write(reinterpret_cast<const char *>(&h), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&b), sizeof(uint8_t));
            ofs.write(reinterpret_cast<const char *>(&base), sizeof(T));
            L.serialize(ofs);
            if (h > 0) {
                G_plus->serialize(ofs);
                G_minus->serialize(ofs);
            }
        }

        void load(std::ifstream &ifs, const std::shared_ptr<IBitVectorFactory> bit_vector_factory) override {
            ifs.read(reinterpret_cast<char *>(&h), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&b), sizeof(uint8_t));
            ifs.read(reinterpret_cast<char *>(&base), sizeof(T));
            L.load(ifs);
            if (h > 0) {
                G_plus = bit_vector_factory->from_stream(ifs);
                G_plus->enable_rank();
                G_plus->enable_select0();

                G_minus = bit_vector_factory->from_stream(ifs);
                G_minus->enable_rank();
                G_minus->enable_select0();
            } else {
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return L.size();
        }

        [[nodiscard]] size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (h > 0) {
                total_bytes += G_plus->size_in_bytes();
                total_bytes += G_minus->size_in_bytes();
            }
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            return total_bytes;
        }

        [[nodiscard]] size_t size_in_bytes_without_supports() const override {
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };
            size_t total_bytes = 0;
            // Raw payload bits only
            total_bytes += sdsl::size_in_bytes(L);
            if (h > 0) {
                total_bytes += bits_to_bytes(G_plus->size());
                total_bytes += bits_to_bytes(G_minus->size());
            }
            // Fixed metadata
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
            
            // G_plus, G_minus bit vectors (if they exist)
            if (h > 0) {
                total_bytes += bits_to_bytes(G_plus->size());
                total_bytes += bits_to_bytes(G_minus->size());
            }
            
            // Fixed metadata
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            
            return total_bytes;
        }

        [[nodiscard]] uint8_t split_point() const override {
            return this->b;
        }
    };
} // namespace gef

#endif
