//
// Created by Michelangelo Pucci on 06/07/25.
//

#ifndef U_GEF_HPP
#define U_GEF_HPP

#include <iostream>
#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits> // Required for std::make_unsigned
#include "IGEF.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"


namespace gef {
    template<typename T>
    class U_GEF : public IGEF<T> {
    private:
        // Bit-vector such that B[i] = 0 <==> 0 <= highPart(i) - highPart(i - 1) <= h
        std::unique_ptr<IBitVector> B;

        /*
         * Bit-vector that store the gaps between consecutive high-parts
         * such that 0 <= highPart(i) - highPart(i - 1) <= h
         */
        std::unique_ptr<IBitVector> G;

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


        static uint8_t optimal_split_point(const std::vector<T> S, const uint8_t total_bits, const T min, const T max) {
            size_t g = 0;
            for (size_t i = 1; i < S.size(); i++) {
                if (S[i] >= S[i - 1])
                    g += S[i] - S[i - 1];
                else
                    g += max - min;
            }
            return ceil(log2(g / S.size()));
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

        ~U_GEF() override = default;

        // Default constructor
        U_GEF() : h(0), b(0), base(0) {
        }

        // 2. Copy Constructor
        U_GEF(const U_GEF &other)
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

                G = other.G->clone();
                G->enable_rank();
                G->enable_select0();
            } else {
                B = nullptr;
                G = nullptr;
            }
        }

        // Friend swap function for copy-and-swap idiom
        friend void swap(U_GEF &first, U_GEF &second) noexcept {
            using std::swap;
            swap(first.B, second.B);
            swap(first.H, second.H);
            swap(first.L, second.L);
            swap(first.h, second.h);
            swap(first.b, second.b);
            swap(first.base, second.base);
            swap(first.G, second.G);
        }

        // 3. Copy Assignment Operator (using copy-and-swap idiom)
        U_GEF &operator=(const U_GEF &other) {
            if (this != &other) {
                U_GEF temp(other);
                swap(*this, temp);
            }
            return *this;
        }

        // 4. Move Constructor
        U_GEF(U_GEF &&other) noexcept
            : IGEF<T>(std::move(other)),
              B(std::move(other.B)),
              G(std::move(other.G)),
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
        U_GEF &operator=(U_GEF &&other) noexcept {
            if (this != &other) {
                B = std::move(other.B);
                G = std::move(other.G);
                H = std::move(other.H);
                L = std::move(other.L);
                h = other.h;
                b = other.b;
                base = other.base;
            }
            return *this;
        }


        // Constructor
        U_GEF(std::shared_ptr<IBitVectorFactory> bit_vector_factory,
              const std::vector<T> &S) {
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


            b = optimal_split_point(S,
                                    total_bits,
                                    min_val,
                                    max_val);
            h = total_bits - b;

            L = sdsl::int_vector<>(S.size(), 0, b);
            if (h == 0) {
                for (size_t i = 0; i < S.size(); i++)
                    L[i] = S[i] - base;
                B = nullptr;
                G = nullptr;
                H.resize(0);
                return;
            }

            B = bit_vector_factory->create(S.size());
            T lastHighBits = 0;
            std::vector<T> tempH;
            tempH.reserve(S.size());
            std::vector<T> tempG;
            tempG.reserve(S.size());
            size_t g_size = 0;
            for (size_t i = 0; i < S.size(); i++) {
                const T element = S[i] - base;
                const T highBits = highPart(element, total_bits, total_bits - b);
                const T lowBits = lowPart(element, b);
                L[i] = lowBits;
                B->set(i, i == 0 || highBits < lastHighBits || highBits >= lastHighBits + h);
                if ((*B)[i] == 1)
                    tempH.push_back(highBits);

                tempG.push_back((1 - (*B)[i]) * (highBits - lastHighBits));
                g_size += (1 - (*B)[i]) * (highBits - lastHighBits) + 1;

                lastHighBits = highBits;
            }
            B->enable_rank();
            B->enable_select1();

            H = sdsl::int_vector<>(tempH.size(), 0, total_bits - b);
            for (size_t i = 0; i < tempH.size(); i++)
                H[i] = tempH[i];

            G = bit_vector_factory->create(g_size);
            size_t pos = 0;
            for (size_t i = 0; i < tempG.size(); i++) {
                auto val = tempG[i];
                for (size_t j = 0; j < val; j++)
                    G->set(pos++, true);
                G->set(pos++, false);
            }

            G->enable_rank();
            G->enable_select0();
        }

        T operator[](size_t index) const override {
            if (h == 0)
                return base + L[index];

            const size_t run_index = B->rank(index + 1);
            const T base_high_val = H[run_index - 1];
            const size_t run_start_pos = B->select(run_index);
            const size_t gap_sum_before_run = run_start_pos > 0 ?
                G->rank(G->select0(run_start_pos + 1)) : 0;
            const size_t total_gap = G->rank(G->select0(index + 1));
            const size_t gap_in_run = total_gap - gap_sum_before_run;

            const T high_val = base_high_val + gap_in_run;

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
                G->serialize(ofs);
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
                G = bit_vector_factory->from_stream(ifs);
                G->enable_rank();
                G->enable_select0();
            } else {
                B = nullptr;
                G = nullptr;
            }
        }

        size_t size() const override {
            return L.size();
        }

        size_t size_in_bytes() const override {
            size_t total_bytes = 0;
            if (B) {
                total_bytes += B->size_in_bytes();
                total_bytes += G->size_in_bytes();
            }
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sdsl::size_in_bytes(H);
            total_bytes += sizeof(base);
            total_bytes += sizeof(h);
            total_bytes += sizeof(b);
            return total_bytes;
        }
    };
} // namespace gef

#endif
