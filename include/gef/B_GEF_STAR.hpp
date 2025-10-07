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
#include "simd_utils.hpp"
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

        static size_t evaluate_space(const GapComputation& gap_computation, const uint8_t b) {
            const size_t N = gap_computation.negative_gaps + gap_computation.positive_gaps;
            return N * (b + 2) + gap_computation.sum_of_negative_gaps + gap_computation.sum_of_positive_gaps;
        }

        static double approximated_optimal_split_point(const std::vector<T> &S, const T min, const T max) {
            const GapComputation gap_computation = simd_optimized_variation_of_original_vec(S, min, max);
            const size_t total_variation = gap_computation.sum_of_negative_gaps + gap_computation.sum_of_positive_gaps;

            return log2(log(2)*total_variation / S.size());
        }

        static std::pair<uint8_t, GapComputation> approximate_optimal_split_point(const std::vector<T> &S,
                                                       const T min, const T max) {
            const double approx_b = approximated_optimal_split_point(S, min, max);
            if (ceil(approx_b) == floor(approx_b))
                return {
                    static_cast<uint8_t>(floor(approx_b)),
                    simd_optimized_variation_of_shifted_vec(S,  min, max, floor(approx_b))
                };

            const uint8_t ceilB = ceil(approx_b);
            const uint8_t floorB = floor(approx_b);
            const std::vector<GapComputation> gap_computation = simd_optimized_total_variation_of_shifted_vec_with_multiple_shifts(S, min, max, floorB, ceilB);
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
            if (best_space > S.size() * total_bits)
                return {total_bits, simd_optimized_variation_of_shifted_vec(S, min, max, total_bits)};
            return {floorB + best_index, gap_computation[best_index]};
        }


        static uint8_t brute_force_optima_split_point(const std::vector<T> &S, const uint8_t total_bits, const T min,
                                                      const T /*max*/, const std::shared_ptr<IBitVectorFactory> &factory) {
            uint8_t best_split_point = 0;
            size_t best_space = std::numeric_limits<size_t>::max();
            for (uint8_t b = 0; b <= total_bits; b++) {
                const size_t space = evaluate_space(S, min, total_bits, b, factory);
                if (space < best_space) {
                    best_split_point = b;
                    best_space = space;
                }
            }
            return best_split_point;
        }

        static std::pair<uint8_t, GapComputation> optimal_split_point(const std::vector<T> &S, const T min, const T max) {
            const double approx_b = approximated_optimal_split_point(S, min, max);

            const uint8_t floorB = floor(approx_b);
            const auto gap_computation = simd_optimized_total_variation_of_shifted_vec_with_multiple_shifts(
                S, min, max, floorB > 0 ? floorB - 1 : floorB, floorB + 3);
            size_t best_index = 0;
            size_t best_space = SIZE_MAX;
            for (size_t i = 0; i < gap_computation.size(); i++) {
                const size_t space = evaluate_space(gap_computation[i], floorB - 1 + i);
                if (space < best_space) {
                    best_index = i;
                    best_space = space;
                }
            }

            const uint64_t u = static_cast<uint64_t>(max) - static_cast<uint64_t>(min) + 1;
            const uint8_t total_bits = (u > 1) ? static_cast<uint8_t>(std::ceil(std::log2(static_cast<double>(u)))) : 1;
            if (best_space > S.size() * total_bits)
                return {total_bits, simd_optimized_variation_of_shifted_vec(S, min, max, total_bits)};
            return {static_cast<uint8_t>(floorB > 0 ? floorB - 1 + best_index : floorB + best_index), gap_computation[best_index]};
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
                     SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT) {
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
            const uint64_t u = max_val - base + 1;
            const uint8_t total_bits = (u > 1) ? static_cast<uint8_t>(floor(log2(u)) + 1) : 1;
            GapComputation gap_computation;

            switch (strategy) {
                case APPROXIMATE_SPLIT_POINT:
                    std::tie(b, gap_computation) = approximate_optimal_split_point(S, base, max_val);
                    break;
                case OPTIMAL_SPLIT_POINT:
                    std::tie(b, gap_computation) = optimal_split_point(S, base, max_val);
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
            T lastHighBits = 0;

            const U low_mask = b ? (U(~U(0)) >> (sizeof(T) * 8 - b)) : U(0);

            if (b == 0) {
                for (size_t i = 0; i < N; ++i) {
                    const T element = S[i] - base;
                    const T currentHighBits = static_cast<T>(static_cast<U>(element) >> 0);

                    const int64_t gap = static_cast<int64_t>(currentHighBits) - static_cast<int64_t>(lastHighBits);

                    if (gap > 0) {
                        G_plus->set_range(g_plus_pos, static_cast<size_t>(gap), true);
                        g_plus_pos += static_cast<size_t>(gap);
                    } else {
                        const size_t neg = static_cast<size_t>(-gap);
                        G_minus->set_range(g_minus_pos, neg, true);
                        g_minus_pos += neg;
                    }
                    // Adding terminators
                    G_minus->set(g_minus_pos++, false);
                    G_plus->set(g_plus_pos++, false);

                    lastHighBits = currentHighBits;
                }
            } else {
                for (size_t i = 0; i < N; ++i) {
                    const T element = S[i] - base;
                    L[i] = static_cast<T>(static_cast<U>(element) & low_mask);

                    const T currentHighBits = static_cast<T>(static_cast<U>(element) >> b);
                    const int64_t gap = static_cast<int64_t>(currentHighBits) - static_cast<int64_t>(lastHighBits);

                    if (gap > 0) {
                        G_plus->set_range(g_plus_pos, static_cast<size_t>(gap), true);
                        g_plus_pos += static_cast<size_t>(gap);
                    } else {
                        // gap <= 0
                        const size_t neg = static_cast<size_t>(-gap);
                        G_minus->set_range(g_minus_pos, neg, true);
                        g_minus_pos += neg;
                    }
                    // Adding terminators
                    G_minus->set(g_minus_pos++, false);
                    G_plus->set(g_plus_pos++, false);

                    lastHighBits = currentHighBits;
                }
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

        [[nodiscard]] uint8_t split_point() const override {
            return this->b;
        }
    };
} // namespace gef

#endif