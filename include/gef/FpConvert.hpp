#pragma once
// Exact conversions between scaled integers v and doubles x = v / 10^p, used by the
// floating-point forks (RLE_GEF_FP, B_STAR_GEF_FP). int -> double returns the correctly
// rounded quotient without a division: q = v * inv, r = fma(-q, scale, v) (exact),
// q + r * inv (one FMA correction). double -> int is round-to-nearest of x * scale.
// Written with intrinsics so that -ffast-math cannot rewrite the arithmetic.
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// Output lines are requested this many 64-value blocks ahead of the stores (0 disables).
#ifndef GEF_FP_PREFETCH
#define GEF_FP_PREFETCH 8
#endif

namespace gef::fp {

inline double scale_for(int64_t decimals) {
    double m = 1;
    for (int64_t d = 0; d < decimals; ++d) m *= 10;
    return m;
}

template <class T>
inline void scaled_to_double(const T* __restrict src, double* __restrict dst, size_t n, double scale, double inv) {
#if defined(GEF_FP_IDENTITY_CONTROL)
    // Measurement control only: same loop and stores, int64 bits copied unconverted.
    (void)scale; (void)inv;
    std::memcpy(dst, src, n * sizeof(T));
    return;
#endif
    size_t i = 0;
#if defined(__AVX512DQ__) && defined(__AVX512VL__) && !defined(GEF_DISABLE_SIMD)
    if constexpr (sizeof(T) == 8) {
#if !defined(GEF_FP_NO_ZMM)
        const __m512d s8 = _mm512_set1_pd(scale), r8 = _mm512_set1_pd(inv);
        for (; i + 8 <= n; i += 8) {
            const __m512d v = _mm512_cvtepi64_pd(_mm512_loadu_si512(src + i));
            const __m512d q = _mm512_mul_pd(v, r8);
            _mm512_storeu_pd(dst + i, _mm512_fmadd_pd(_mm512_fnmadd_pd(q, s8, v), r8, q));
        }
#endif
        const __m256d s = _mm256_set1_pd(scale), r = _mm256_set1_pd(inv);
        for (; i + 4 <= n; i += 4) {
            const __m256d v = _mm256_cvtepi64_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i)));
            const __m256d q = _mm256_mul_pd(v, r);
            _mm256_storeu_pd(dst + i, _mm256_fmadd_pd(_mm256_fnmadd_pd(q, s, v), r, q));
        }
    }
#endif
    for (; i < n; ++i) {
#if defined(__FMA__)
        const __m128d v = _mm_set_sd(static_cast<double>(src[i]));
        const __m128d q = _mm_mul_sd(v, _mm_set_sd(inv));
        dst[i] = _mm_cvtsd_f64(_mm_fmadd_sd(_mm_fnmadd_sd(q, _mm_set_sd(scale), v), _mm_set_sd(inv), q));
#else
        // Without FMA intrinsics (e.g. non-x86), divide: correctly rounded by definition
        // as long as this code is not compiled with reciprocal approximations.
        dst[i] = static_cast<double>(src[i]) / scale;
#endif
    }
}

template <class T>
inline T scaled_to_double_one(T v, double scale, double inv) = delete;

inline double scaled_to_double_one(int64_t v, double scale, double inv) {
    double out;
    scaled_to_double(&v, &out, 1, scale, inv);
    return out;
}

// Converts like scaled_to_double, requesting the output lines GEF_FP_PREFETCH_VALUES
// values ahead for writing (bounded by dst_end), so that a burst of stores from an
// L1 buffer does not wait for each line to arrive from memory.
#ifndef GEF_FP_PREFETCH_VALUES
#define GEF_FP_PREFETCH_VALUES 512
#endif
template <class T>
inline void scaled_to_double_prefetch(const T* __restrict src, double* __restrict dst, size_t n, double scale, double inv,
                                      const double* dst_end) {
    for (size_t done = 0; done < n; done += 64) {
        const size_t len = n - done < 64 ? n - done : 64;
        const double* ahead = dst + done + GEF_FP_PREFETCH_VALUES;
        if (ahead + 64 <= dst_end)
            for (int line = 0; line < 8; ++line) __builtin_prefetch(reinterpret_cast<const char*>(ahead) + line * 64, 1, 3);
        scaled_to_double(src + done, dst + done, len, scale, inv);
    }
}

template <class T>
inline void double_to_scaled(const double* __restrict src, T* __restrict dst, size_t n, double scale) {
    size_t i = 0;
#if defined(__AVX512DQ__) && defined(__AVX512VL__) && !defined(GEF_DISABLE_SIMD)
    if constexpr (sizeof(T) == 8) {
        const __m256d s = _mm256_set1_pd(scale);
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                                _mm256_cvtpd_epi64(_mm256_mul_pd(_mm256_loadu_pd(src + i), s)));
    }
#endif
    for (; i < n; ++i) {
#if defined(__SSE2__)
        dst[i] = static_cast<T>(_mm_cvtsd_si64(_mm_mul_sd(_mm_set_sd(src[i]), _mm_set_sd(scale))));
#else
        dst[i] = static_cast<T>(std::llrint(src[i] * scale));
#endif
    }
}

} // namespace gef::fp
