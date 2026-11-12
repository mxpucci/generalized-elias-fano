#pragma once
// Exact conversions between scaled integers v and doubles x = v / 10^p, used by the
// floating-point forks (RLE_GEF_FP, B_STAR_GEF_FP).
//
// int -> double returns the correctly rounded quotient without a division:
//   default : q = v * inv, r = fma(-q, scale, v) (exact), q + r * inv (one FMA correction);
//   two_op  : fma(v, inv, v * lo) with inv + lo ~ 1 / scale, cheaper but not exact for
//             every input: the forks enable it per partition only after checking at
//             compression that it reproduces every value of that partition.
// double -> int is round-to-nearest of x * scale.
// Written with intrinsics so that -ffast-math cannot change the arithmetic.
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

// Requests of at least this many values (1 MB of doubles) are assumed not to be
// cache-resident: the decoders then request output lines GEF_FP_PREFETCH blocks ahead.
#ifndef GEF_FP_PREFETCH_MIN_VALUES
#define GEF_FP_PREFETCH_MIN_VALUES 131072
#endif

#if defined(__AVX512DQ__) && defined(__AVX512VL__) && !defined(GEF_DISABLE_SIMD)
#define GEF_FP_VECTOR 1
#endif

namespace gef::fp {

inline double scale_for(int64_t decimals) {
    double m = 1;
    for (int64_t d = 0; d < decimals; ++d) m *= 10;
    return m;
}

// lo = RN((1 - inv * scale) / scale), the low part of 1 / scale.
inline double low_part(double scale, double inv) {
#if defined(__FMA__)
    return _mm_cvtsd_f64(_mm_fnmadd_sd(_mm_set_sd(inv), _mm_set_sd(scale), _mm_set_sd(1.0))) / scale;
#else
    return 0.0 * inv;
#endif
}

// Conversion constants for one partition, broadcast once for the decoders' store sites.
struct Scale {
    double scale, inv, lo;
    bool stream = false; // output too large for the caches: prefetch output lines ahead
    bool two_op = false; // two-operation formula verified for this partition
#if defined(GEF_FP_VECTOR)
    __m512d scale8, inv8, lo8;
    __m256d scale4, inv4, lo4;
#endif
    Scale(double s, double i, bool streaming = false, bool two = false)
        : scale(s), inv(i), lo(low_part(s, i)), stream(streaming), two_op(two) {
#if defined(GEF_FP_VECTOR)
        scale8 = _mm512_set1_pd(s); inv8 = _mm512_set1_pd(i); lo8 = _mm512_set1_pd(lo);
        scale4 = _mm256_set1_pd(s); inv4 = _mm256_set1_pd(i); lo4 = _mm256_set1_pd(lo);
#endif
    }
};

// One scaled integer -> double.
inline double one(int64_t v, double scale, double inv, double lo, bool two_op) {
#if defined(GEF_FP_IDENTITY_CONTROL)
    // Measurement control only: the int64 bits, unconverted.
    double bits;
    std::memcpy(&bits, &v, sizeof bits);
    (void)scale; (void)inv; (void)lo; (void)two_op;
    return bits;
#elif defined(__FMA__)
    const __m128d x = _mm_set_sd(static_cast<double>(v));
    if (two_op) return _mm_cvtsd_f64(_mm_fmadd_sd(x, _mm_set_sd(inv), _mm_mul_sd(x, _mm_set_sd(lo))));
    const __m128d q = _mm_mul_sd(x, _mm_set_sd(inv));
    return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_fnmadd_sd(q, _mm_set_sd(scale), x), _mm_set_sd(inv), q));
#else
    // Without FMA (e.g. non-x86), divide: correctly rounded unless compiled with
    // reciprocal approximations.
    (void)inv; (void)lo; (void)two_op;
    return static_cast<double>(v) / scale;
#endif
}
inline double one(int64_t v, const Scale& k) { return one(v, k.scale, k.inv, k.lo, k.two_op); }

#if defined(GEF_FP_VECTOR)
inline __m512d convert8(__m512i v, const Scale& k) {
    const __m512d x = _mm512_cvtepi64_pd(v);
    if (k.two_op) return _mm512_fmadd_pd(x, k.inv8, _mm512_mul_pd(x, k.lo8));
    const __m512d q = _mm512_mul_pd(x, k.inv8);
    return _mm512_fmadd_pd(_mm512_fnmadd_pd(q, k.scale8, x), k.inv8, q);
}
inline __m256d convert4(__m256i v, const Scale& k) {
    const __m256d x = _mm256_cvtepi64_pd(v);
    if (k.two_op) return _mm256_fmadd_pd(x, k.inv4, _mm256_mul_pd(x, k.lo4));
    const __m256d q = _mm256_mul_pd(x, k.inv4);
    return _mm256_fmadd_pd(_mm256_fnmadd_pd(q, k.scale4, x), k.inv4, q);
}
#endif

// Scaled integers in memory -> doubles.
template <class T>
inline void scaled_to_double(const T* __restrict src, double* __restrict dst, size_t n, const Scale& k) {
#if defined(GEF_FP_IDENTITY_CONTROL)
    std::memcpy(dst, src, n * sizeof(T)); // measurement control only
    (void)k;
    return;
#endif
    size_t i = 0;
#if defined(GEF_FP_VECTOR)
    if constexpr (sizeof(T) == 8) {
        for (; i + 8 <= n; i += 8) _mm512_storeu_pd(dst + i, convert8(_mm512_loadu_si512(src + i), k));
        for (; i + 4 <= n; i += 4)
            _mm256_storeu_pd(dst + i, convert4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i)), k));
    }
#endif
    for (; i < n; ++i) dst[i] = one(static_cast<int64_t>(src[i]), k);
}

#if defined(__AVX512F__)
// Eight scaled integers in a register -> eight doubles stored at dst.
inline void store_i64x8(double* dst, __m512i v, const Scale& k) {
#if defined(GEF_FP_IDENTITY_CONTROL)
    _mm512_storeu_si512(dst, v); // measurement control only
    (void)k;
#elif defined(GEF_FP_VECTOR)
    _mm512_storeu_pd(dst, convert8(v, k));
#else
    alignas(64) int64_t tmp[8];
    _mm512_store_si512(tmp, v);
    scaled_to_double(tmp, dst, 8, k);
#endif
}
#endif

#if defined(__AVX2__)
// Four scaled integers in a register -> four doubles stored at dst.
inline void store_i64x4(double* dst, __m256i v, const Scale& k) {
#if defined(GEF_FP_IDENTITY_CONTROL)
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), v); // measurement control only
    (void)k;
#elif defined(GEF_FP_VECTOR)
    _mm256_storeu_pd(dst, convert4(v, k));
#else
    alignas(32) int64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v);
    scaled_to_double(tmp, dst, 4, k);
#endif
}
#endif

// Whether the two-operation formula reproduces x[i] exactly from v[i] for every i < n.
template <class T>
inline bool two_op_exact(const T* v, const double* x, size_t n, double scale, double inv) {
    const double lo = low_part(scale, inv);
    for (size_t i = 0; i < n; ++i) {
        const double y = one(static_cast<int64_t>(v[i]), scale, inv, lo, true);
        if (std::memcmp(&y, x + i, sizeof y) != 0) return false;
    }
    return true;
}

// Doubles -> scaled integers: round-to-nearest of x * scale.
template <class T>
inline void double_to_scaled(const double* __restrict src, T* __restrict dst, size_t n, double scale) {
    size_t i = 0;
#if defined(GEF_FP_VECTOR)
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
