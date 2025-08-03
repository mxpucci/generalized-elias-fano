//
// Created by Michelangelo Pucci on 25/07/25.
//

#ifndef B_GEF_HPP
#define B_GEF_HPP

#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits>
#include "IGEF.hpp"
#include "RLE_GEF.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"


namespace gef {
    template<typename T>
    class B_GEF : public IGEF<T> {
    private:
        // Bit-vector such that B[i] = 0 <==> 0 <= highPart(i) - highPart(i - 1) <= h
        std::unique_ptr<IBitVector> B;

        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that 0 <= highPart(i) - highPart(i - 1) <= h
         */
        std::unique_ptr<IBitVector> G_plus;


        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that 0 <= highPart(i - 1) - highPart(i) <= h
         */
        std::unique_ptr<IBitVector> G_minus;

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

        static size_t evaluate_space(const std::vector<T> &S, const uint8_t total_bits, uint8_t b) {
            const size_t N = S.size();
            if (N == 0) {
                return sizeof(T) + sizeof(uint8_t) * 2; // Overhead for base, h, b
            }

            // Handle edge cases for the split point 'b'
            if (b == 0) {
                return std::numeric_limits<size_t>::max();
            }
            if (b >= total_bits) {
                // In this case, only the L vector exists, storing the full values (h=0).
                size_t bits = N * total_bits;
                return (bits + 7) / 8 + sizeof(T) + sizeof(uint8_t) * 2;
            }

            const T base = *std::min_element(S.begin(), S.end());
            const uint8_t h = total_bits - b;

            size_t num_exceptions = 0;
            size_t g_unary_bits = 0;
            T lastHighBits = 0;

            // 1. Simulate the creation of B, H, and G to find their required sizes
            for (size_t i = 0; i < N; ++i) {
                const T element = S[i] - base;
                const T currentHighBits = highPart(element, total_bits, h);

                const bool is_exception = (i == 0 || currentHighBits + h <= lastHighBits || currentHighBits >= lastHighBits +
                                           h);

                if (is_exception) {
                    num_exceptions++;
                } else {
                    // This is a regular gap; add its size to G's unary encoded part
                    g_unary_bits += currentHighBits >= lastHighBits ? currentHighBits - lastHighBits : lastHighBits - currentHighBits;
                }
                lastHighBits = currentHighBits;
            }

            // 2. Calculate the total size in bits for all data structures
            const size_t L_bits = N * b; // L stores low bits for all N elements
            const size_t B_bits = N; // B has one bit per element
            const size_t H_bits = num_exceptions * h; // H stores h bits for each exception
            const size_t G_bits = g_unary_bits + 2 * N; // G stores unary gaps + N terminators

            const size_t total_data_bits = L_bits + B_bits + H_bits + G_bits;

            return total_data_bits;
        }

        static uint8_t binary_search_optimal_split_point(const std::vector<T> &S, const uint8_t total_bits, const T /*min*/,
                                          const T /*max*/) {
            if (total_bits <= 1) {
                // Handle trivial cases where a search is not possible.
                size_t space0 = evaluate_space(S, total_bits, 0);
                if (total_bits == 0) return 0;
                size_t space1 = evaluate_space(S, total_bits, 1);
                return (space0 < space1) ? 0 : 1;
            }

            uint8_t lo = 0, hi = total_bits;

            // Golden ratio constant, used to determine the probe points.
            // We use the reciprocal (1/phi) for interval reduction.
            const double inv_phi = (std::sqrt(5.0) - 1.0) / 2.0; // approx 0.618

            // Calculate the initial two interior points.
            uint8_t c = lo + static_cast<uint8_t>(std::round((hi - lo) * (1.0 - inv_phi)));
            uint8_t d = lo + static_cast<uint8_t>(std::round((hi - lo) * inv_phi));

            // Initial evaluations for the two points.
            size_t space_c = evaluate_space(S, total_bits, c);
            size_t space_d = evaluate_space(S, total_bits, d);

            while (c < d) {
                if (space_c < space_d) {
                    // The minimum is in the lower interval [lo, d].
                    // The old 'c' becomes the new 'd'.
                    hi = d - 1;
                    d = c;
                    space_d = space_c;

                    // We only need to compute a new 'c'.
                    c = lo + static_cast<uint8_t>(std::round((hi - lo) * (1.0 - inv_phi)));
                    space_c = evaluate_space(S, total_bits, c);
                } else {
                    // The minimum is in the upper interval [c, hi].
                    // The old 'd' becomes the new 'c'.
                    lo = c + 1;
                    c = d;
                    space_c = space_d;

                    // We only need to compute a new 'd'.
                    d = lo + static_cast<uint8_t>(std::round((hi - lo) * inv_phi));
                    space_d = evaluate_space(S, total_bits, d);
                }
            }

            // The minimum is at lo (or hi, which will be the same).
            return lo;
        }

        static uint8_t approximate_optimal_split_point(const std::vector<T> &S, const uint8_t total_bits, const T min,
                                                        const T max) {
            if (S.size() <= 1) {
                return 0;
            }
            size_t g = 0;
            for (size_t i = 1; i < S.size(); i++) {
                g += S[i] >= S[i - 1] ? S[i] - S[i - 1] : S[i - 1] - S[i];
            }
            double avg_gap = static_cast<double>(g) / S.size();
            if (avg_gap <= 0) {
                return 0;
            }
            return ceil(log2(avg_gap));
        }


        static uint8_t brute_force_optima_split_point(const std::vector<T> &S, const uint8_t total_bits, const T min, const T max) {
            uint8_t best_split_point = 0;
            size_t best_space = evaluate_space(S, total_bits, best_split_point);
            for (uint8_t b = 1; b <= total_bits; b++) {
                const size_t space = evaluate_space(S, total_bits, b);
                if (space < best_space) {
                    best_split_point = b;
                    best_space = space;
                }
            }
            return best_split_point;
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

        ~B_GEF() override = default;

        // Default constructor
        B_GEF() : h(0), b(0), base(0) {
        }

        // 2. Copy Constructor
        B_GEF(const B_GEF &other)
            : IGEF<T>(other), // Slicing is not an issue here as IGEF has no data
              H(other.H),
              L(other.L),
              h(other.h),
              b(other.b),
              base(other.base) {
            if (other.h > 0) {
                B = other.B->clone();
                B->enable_rank();
                B->enable_select1();

                G_plus = other.G_plus->clone();
                G_plus->enable_rank();
                G_plus->enable_select0();

                G_minus = other.G_minus->clone();
                G_minus->enable_rank();
                G_minus->enable_select0();
            } else {
                B = nullptr;
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        // Friend swap function for copy-and-swap idiom
        friend void swap(B_GEF &first, B_GEF &second) noexcept {
            using std::swap;
            swap(first.B, second.B);
            swap(first.H, second.H);
            swap(first.L, second.L);
            swap(first.h, second.h);
            swap(first.b, second.b);
            swap(first.base, second.base);
            swap(first.G_plus, second.G_plus);
            swap(first.G_minus, second.G_minus);
        }

        // 3. Copy Assignment Operator (using copy-and-swap idiom)
        B_GEF &operator=(const B_GEF &other) {
            if (this != &other) {
                B_GEF temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // 4. Move Constructor
        B_GEF(B_GEF &&other) noexcept
            : IGEF<T>(std::move(other)),
              B(std::move(other.B)),
              G_plus(std::move(other.G_plus)),
                G_minus(std::move(other.G_minus)),
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
        B_GEF &operator=(B_GEF &&other) noexcept {
            if (this != &other) {
                B = std::move(other.B);
                G_plus = std::move(other.G_plus);
                G_minus = std::move(other.G_minus);
                H = std::move(other.H);
                L = std::move(other.L);
                h = other.h;
                b = other.b;
                base = other.base;
            }
            return *this;
        }


        // Constructor
        B_GEF(const std::shared_ptr<IBitVectorFactory>& bit_vector_factory,
              const std::vector<T> &S,
              SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT) {
            if (S.empty()) {
                b = 0;
                h = 0;
                base = T{};
                B = nullptr;
                return;
            }

            base = *std::min_element(S.begin(), S.end());
            const int64_t max_val = *std::max_element(S.begin(), S.end());
            const int64_t min_val = base;
            const uint64_t u = max_val - min_val + 1;
            const uint8_t total_bits = (u > 1) ? static_cast<uint8_t>(floor(log2(u)) + 1) : 1;


            switch (strategy) {
                case BINARY_SEARCH_SPLIT_POINT:
                    b = binary_search_optimal_split_point(S, total_bits, min_val, max_val);
                    break;
                case APPROXIMATE_SPLIT_POINT:
                    b = approximate_optimal_split_point(S, total_bits, min_val, max_val);
                    break;
                case BRUTE_FORCE_SPLIT_POINT:
                    b = brute_force_optima_split_point(S, total_bits, min_val, max_val);
                    break;
            }
            h = total_bits - b;

            L = sdsl::int_vector<>(S.size(), 0, b);
            if (h == 0) {
                for (size_t i = 0; i < S.size(); i++)
                    L[i] = S[i] - base;
                B = nullptr;
                G_plus = nullptr;
                G_minus = nullptr;
                H.resize(0);
                return;
            }

            B = bit_vector_factory->create(S.size());
            T lastHighBits = 0;
            std::vector<T> tempH;
            tempH.reserve(S.size());

            std::vector<T> tempGPlus;
            tempGPlus.reserve(S.size());
            std::vector<T> tempGMinus;
            tempGMinus.reserve(S.size());

            size_t g_plus_size = 0;
            size_t g_minus_size = 0;

            for (size_t i = 0; i < S.size(); i++) {
                const T element = S[i] - base;
                const T highBits = highPart(element, total_bits, total_bits - b);
                const T lowBits = lowPart(element, b);
                L[i] = lowBits;

                const int64_t gap = static_cast<int64_t>(highBits) - static_cast<int64_t>(lastHighBits);

                B->set(i, i == 0 || std::abs(gap) > h);

                if ((*B)[i] == 1) {
                    tempH.push_back(highBits);
                }

                if ((*B)[i] == 0 && gap > 0) {
                    tempGPlus.push_back(gap);
                    tempGMinus.push_back(0);
                    g_plus_size += gap;
                } else if ((*B)[i] == 0 && gap <= 0) { // gap <= 0
                    tempGPlus.push_back(0);
                    tempGMinus.push_back(-gap); // -gap is the magnitude
                    g_minus_size += -gap;
                } else {
                    tempGPlus.push_back(0);
                    tempGMinus.push_back(0);
                }

                // Appending trailing zeros
                g_plus_size++;
                g_minus_size++;

                lastHighBits = highBits;
            }
            B->enable_rank();
            B->enable_select1();

            H = sdsl::int_vector<>(tempH.size(), 0, total_bits - b);
            for (size_t i = 0; i < tempH.size(); i++)
                H[i] = tempH[i];

            G_plus = bit_vector_factory->create(g_plus_size);
            size_t pos = 0;
            for (size_t i = 0; i < tempGPlus.size(); i++) {
                auto val = tempGPlus[i];
                for (size_t j = 0; j < val; j++)
                    G_plus->set(pos++, true);
                G_plus->set(pos++, false);
            }
            G_plus->enable_rank();
            G_plus->enable_select0();

            G_minus = bit_vector_factory->create(g_minus_size);
            pos = 0;
            for (size_t i = 0; i < tempGMinus.size(); i++) {
                auto val = tempGMinus[i];
                for (size_t j = 0; j < val; j++)
                    G_minus->set(pos++, true);
                G_minus->set(pos++, false);
            }
            G_minus->enable_rank();
            G_minus->enable_select0();
        }

        T operator[](size_t index) const override {
            if (h == 0)
                return base + L[index];

            const size_t run_index = B->rank(index + 1);
            const T base_high_val = H[run_index - 1];
            const size_t run_start_pos = B->select(run_index);

            const size_t pos_gap_sum_before_run = run_start_pos > 0 ? G_plus->rank(G_plus->select0(run_start_pos + 1)) : 0;
            const size_t total_pos_gap = G_plus->rank(G_plus->select0(index + 1));
            const size_t pos_gap_in_run = total_pos_gap - pos_gap_sum_before_run;

            const size_t neg_gap_sum_before_run = run_start_pos > 0 ? G_minus->rank(G_minus->select0(run_start_pos + 1)) : 0;
            const size_t total_neg_gap = G_minus->rank(G_minus->select0(index + 1));
            const size_t neg_gap_in_run = total_neg_gap - neg_gap_sum_before_run;

            const T high_val = base_high_val + pos_gap_in_run - neg_gap_in_run;

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
            H.serialize(ofs);
            if (h > 0) {
                B->serialize(ofs);
                G_plus->serialize(ofs);
                G_minus->serialize(ofs);
            }
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
                B->enable_select1();
                G_plus = bit_vector_factory->from_stream(ifs);
                G_plus->enable_rank();
                G_plus->enable_select0();

                G_minus = bit_vector_factory->from_stream(ifs);
                G_minus->enable_rank();
                G_minus->enable_select0();
            } else {
                B = nullptr;
                G_plus = nullptr;
                G_minus = nullptr;
            }
        }

        [[nodiscard]] size_t size() const override {
            return L.size();
        }

        [[nodiscard]] size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (B) {
                total_bytes += B->size_in_bytes();
                total_bytes += G_plus->size_in_bytes();
                total_bytes += G_minus->size_in_bytes();
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
