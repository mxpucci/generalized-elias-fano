// Reference conversion for bench_fp_conversion, compiled without -ffast-math.
#include <cstdint>
double strict_divide(int64_t v, double m) { return static_cast<double>(v) / m; }
