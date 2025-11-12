//
// Created by Michelangelo Pucci on 06/07/25.
//

#ifndef U_GEF_HPP
#define U_GEF_HPP

#include "gap_computation_utils.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <memory>
#include <filesystem>
#include "sdsl/int_vector.hpp"
#include <vector>
#include <type_traits> // Required for std::make_unsigned
#include "IGEF.hpp"
#include "RLE_GEF.hpp"
#include "../datastructures/IBitVector.hpp"
#include "../datastructures/IBitVectorFactory.hpp"
#include "../datastructures/SDSLBitVectorFactory.hpp"

#if (defined(__AVX2__) || defined(__SSE4_2__)) && !defined(GEF_DISABLE_SIMD)
#include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(GEF_DISABLE_SIMD)
#include <arm_neon.h>
#endif

namespace gef {
    template<typename T>
    class U_GEF : public IGEF<T> {
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
            // Clamp to the size of T to prevent undefined behavior in shifts
            return static_cast<uint8_t>(std::min<size_t>(bits, sizeof(T) * 8));
        }
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

        static size_t evaluate_space(const size_t N,
                                     const size_t total_bits,
                                     const uint8_t b,
                                     const GapComputation &gc) {
            if (N == 0)
                return sizeof(T) + 2;  // Just metadata

            if (b >= total_bits) {
                // All in L, no high bits
                size_t l_bytes = ((N * static_cast<size_t>(total_bits) + 7) / 8);
                return l_bytes + sizeof(T) + 2;
            }

            // Exceptions
            const auto exceptions = gc.positive_exceptions_count + gc.negative_exceptions_count;
            if (exceptions == 0 || exceptions > N) {
                // Invalid - fallback
                size_t fallback_bits = N * static_cast<size_t>(total_bits);
                return ((fallback_bits + 7) / 8) + sizeof(T) + 2;
            }
            const size_t non_exc = N - exceptions;

            // Helper to convert bits to bytes
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };

            // Calculate components in bytes (theoretical)
            size_t l_bytes = bits_to_bytes(N * b);
            size_t h_bytes = bits_to_bytes(exceptions * static_cast<size_t>(total_bits - b));
            size_t b_bits = N;
            size_t b_bytes = bits_to_bytes(b_bits);
            size_t g_bits = gc.sum_of_positive_gaps_without_exception + non_exc;
            size_t g_bytes = bits_to_bytes(g_bits);

            // Metadata
            size_t metadata = sizeof(T) + sizeof(uint8_t) + sizeof(uint8_t);

            return l_bytes + h_bytes + b_bytes + g_bytes + metadata;
        }

        static std::pair<uint8_t, GapComputation> approximate_optimal_split_point
        (const std::vector<T> &S,
            const T min,
                                                       const T max) {
            const size_t total_bits = bits_for_range(min, max);

            const auto gc = variation_of_original_vec(S, min, max);
            const auto total_variation = gc.sum_of_positive_gaps;

            const double approximated_split_point = log2(total_variation / S.size());
            // Clamp candidate b to [0, total_bits]
            auto clamp_b = [&](double x) -> uint8_t {
                long long bi = static_cast<long long>(std::llround(x));
                long long lo = 0;
                long long hi = static_cast<long long>(total_bits);
                if (bi < lo) bi = lo;
                if (bi > hi) bi = hi;
                return static_cast<uint8_t>(bi);
            };
            uint8_t bFloor = clamp_b(approximated_split_point);
            uint8_t bCeil = clamp_b(approximated_split_point + 1);

            const uint8_t min_b = std::min(bFloor, bCeil);
            const uint8_t max_b = std::max(bFloor, bCeil);

            // Only compute gap computations for the 2 candidates we actually need
            const auto gcs = total_variation_of_shifted_vec_with_multiple_shifts(
                S, min, max, min_b, max_b, ExceptionRule::UGEF);

            size_t best_index = 0;
            size_t best_space = evaluate_space(S.size(), total_bits, min_b, gcs[0]);
            const auto tmpSpace = evaluate_space(S.size(), total_bits, max_b, gcs[gcs.size() - 1]);
            if (tmpSpace < best_space) {
                best_index = gcs.size() - 1;
            }

            return {static_cast<uint8_t>(min_b + best_index), gcs[best_index]};
        }

        static std::pair<uint8_t, GapComputation> brute_force_optima_split_point(
            const std::vector<T> &S,
            const T min_val,
            const T max_val) {
            const size_t N = S.size();
            if (N == 0) return {0u, GapComputation{}};

            const uint8_t total_bits = bits_for_range(min_val, max_val);

            // Compute all gap computations in a single pass
            const auto gcs = compute_all_gap_computations(S, min_val, max_val, ExceptionRule::UGEF, total_bits);

            uint8_t best_b = total_bits;
            size_t best_space = evaluate_space(N, total_bits, total_bits, GapComputation{});

            // Check all b values for true optimality
            for (uint8_t b = 1; b < total_bits; ++b) {
                const size_t space = evaluate_space(N, total_bits, b, gcs[b]);
                if (space < best_space) {
                    best_b = b;
                    best_space = space;
                }
            }
            
            // Return the gap computation for the selected split point
            if (best_b < total_bits) {
                return {best_b, gcs[best_b]};
            } else {
                return {total_bits, GapComputation{}};
            }
        }

        static std::make_unsigned_t<T>
        highPart(const std::make_unsigned_t<T> x, const uint8_t total_bits, const uint8_t highBits) {
            const uint8_t lowBits = total_bits - highBits;
            if (lowBits >= sizeof(T) * 8) return 0;
            return static_cast<std::make_unsigned_t<T>>(x >> lowBits);
        }

        static std::make_unsigned_t<T>
        lowPart(const std::make_unsigned_t<T> x, const uint8_t lowBits) {
            if (lowBits >= sizeof(T) * 8) return x;
            const std::make_unsigned_t<T> mask = (static_cast<std::make_unsigned_t<T>>(1) << lowBits) - 1;
            return static_cast<std::make_unsigned_t<T>>(x & mask);
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
        U_GEF(const std::shared_ptr<IBitVectorFactory> &bit_vector_factory,
              const std::vector<T> &S,
              SplitPointStrategy strategy = APPROXIMATE_SPLIT_POINT) {
            const size_t N = S.size();
            if (N == 0) {
                b = 0;
                h = 0;
                base = T{};
                B = nullptr;
                return;
            }

            auto [min_it, max_it] = std::minmax_element(S.begin(), S.end());
            base = *min_it;
            const T max_val = *max_it;

            const uint8_t total_bits = bits_for_range(base, max_val);
            GapComputation gc{};
            if (strategy == OPTIMAL_SPLIT_POINT) {
                std::tie(b, gc) = brute_force_optima_split_point(S, base, max_val);
            } else {
                std::tie(b, gc) = approximate_optimal_split_point(S, base, max_val);
            }
            h = (b >= total_bits) ? 0 : static_cast<uint8_t>(total_bits - b);

            L = sdsl::int_vector<>(N, 0, b);

            if (h == 0) {
                // All in L
                for (size_t i = 0; i < N; ++i) {
                    L[i] = S[i] - base;
                }
                B = nullptr;
                G = nullptr;
                H.resize(0);
                return;
            }

            // Use precomputed gc for allocation
            const size_t exceptions = gc.positive_exceptions_count + gc.negative_exceptions_count;
            const size_t non_exceptions = N - exceptions;
            const size_t g_bits = gc.sum_of_positive_gaps_without_exception + non_exceptions;

            B = bit_vector_factory->create(N);
            H = sdsl::int_vector<>(exceptions, 0, h);
            G = bit_vector_factory->create(g_bits);

            // Single pass to populate all structures
            size_t h_idx = 0;
            size_t g_pos = 0;
            using U = std::make_unsigned_t<T>;
            U lastHighBits = 0;
            const uint64_t h_u64 = static_cast<uint64_t>(h);
            
            // Precompute low mask once (same optimization as B_GEF_STAR)
            const U low_mask = b < sizeof(T) * 8 ? ((U(1) << b) - 1) : U(~U(0));

            for (size_t i = 0; i < N; ++i) {
                const U element_u = static_cast<U>(S[i]) - static_cast<U>(base);
                L[i] = static_cast<typename sdsl::int_vector<>::value_type>(element_u & low_mask);
                
                const U current_high_part = (b < sizeof(T) * 8) ? (element_u >> b) : U(0);
                const int64_t gap = static_cast<int64_t>(current_high_part) - static_cast<int64_t>(lastHighBits);
                
                // Exception check: i==0 or gap<0 or gap>=h
                const bool is_exception = (i == 0) | (gap < 0) | (static_cast<uint64_t>(gap) >= h_u64);

                if (is_exception) [[unlikely]] {
                    B->set(i, true);
                    H[h_idx++] = current_high_part;
                } else [[likely]] {
                    B->set(i, false);
                    const uint64_t g = static_cast<uint64_t>(gap);
                    G->set_range(g_pos, g, true);
                    g_pos += g;
                    G->set(g_pos++, false);
                }
                lastHighBits = current_high_part;
            }

            assert(g_pos == g_bits);

            // Enable rank/select support
            B->enable_rank();
            B->enable_select1();
            G->enable_rank();
            G->enable_select0();
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
            
            // Optimized range access with incremental rank computation
            using U = std::make_unsigned_t<T>;
            
            size_t current_rank = B->rank(startIndex + 1);
            T base_high_val = H[current_rank - 1];
            size_t run_start_pos = B->select(current_rank);
            size_t zeros_before_run = (run_start_pos + 1) - current_rank;
            
            for (size_t i = startIndex; i < endIndex; ++i) {
                const bool is_exception = (*B)[i];
                
                if (is_exception) [[unlikely]] {
                    // Update tracking for new run
                    if (i != startIndex) {
                        current_rank = B->rank(i + 1);
                        base_high_val = H[current_rank - 1];
                        run_start_pos = B->select(current_rank);
                        zeros_before_run = (run_start_pos + 1) - current_rank;
                    }
                    result.push_back(base + (L[i] | (base_high_val << b)));
                } else [[likely]] {
                    const size_t zero_rank_at_i = (i + 1) - current_rank;
                    
                    size_t gap_in_run;
                    if (zeros_before_run == 0) [[unlikely]] {
                        gap_in_run = G->rank(G->select0(zero_rank_at_i));
                    } else [[likely]] {
                        const size_t total_gap_sum = G->rank(G->select0(zero_rank_at_i));
                        const size_t gap_before_run = G->rank(G->select0(zeros_before_run));
                        gap_in_run = total_gap_sum - gap_before_run;
                    }
                    
                    const T high_val = base_high_val + static_cast<T>(gap_in_run);
                    result.push_back(base + (L[i] | (high_val << b)));
                }
            }
            
            return result;
        }

        T operator[](size_t index) const override {
            // Case 1: No high bits are used (h=0).
            // All information is stored in the L vector. Reconstruction is trivial.
            if (h == 0) {
                return base + L[index];
            }

            // Find the number of exceptions up to and including 'index'.
            const size_t run_index = B->rank(index + 1);
            const T base_high_val = H[run_index - 1];
            
            // Fast path: check if this is an exception
            const bool is_exception = (*B)[index];
            if (is_exception) [[unlikely]] {
                // Exception: high part stored explicitly, just combine with low part
                return base + (L[index] | (base_high_val << b));
            }
            
            // Non-exception: reconstruct from gaps
            const size_t run_start_pos = B->select(run_index);
            const size_t zero_rank_at_index = (index + 1) - run_index;
            const size_t zeros_before_run = (run_start_pos + 1) - run_index;

            // Optimize common case: first element in run
            size_t gap_in_run;
            if (zeros_before_run == 0) [[unlikely]] {
                gap_in_run = G->rank(G->select0(zero_rank_at_index));
            } else [[likely]] {
                const size_t total_gap_sum = G->rank(G->select0(zero_rank_at_index));
                const size_t gap_before_run = G->rank(G->select0(zeros_before_run));
                gap_in_run = total_gap_sum - gap_before_run;
            }

            const T high_val = base_high_val + static_cast<T>(gap_in_run);
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

        [[nodiscard]] size_t size() const override {
            return L.size();
        }

        [[nodiscard]] size_t size_in_bytes() const override {
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

        [[nodiscard]] size_t size_in_bytes_without_supports() const override {
            auto bits_to_bytes = [](size_t bits) -> size_t { return (bits + 7) / 8; };
            size_t total_bytes = 0;
            // Raw payload bits only
            total_bytes += sdsl::size_in_bytes(L);
            total_bytes += sdsl::size_in_bytes(H);
            if (B) {
                total_bytes += bits_to_bytes(B->size());
            }
            if (G) {
                total_bytes += bits_to_bytes(G->size());
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
            
            // H vector: use width * size formula (theoretical)
            total_bytes += bits_to_bytes(H.size() * H.width());
            
            // B, G bit vectors (if they exist)
            if (B) {
                total_bytes += bits_to_bytes(B->size());
            }
            if (G) {
                total_bytes += bits_to_bytes(G->size());
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
