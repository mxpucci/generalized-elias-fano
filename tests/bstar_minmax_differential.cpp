// Standalone baseline/candidate differential driver. Compile against each include
// tree and compare stdout byte-for-byte; a mismatch or roundtrip error fails.
#include "gef/B_STAR_GEF.hpp"
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

static uint64_t digest = 1469598103934665603ULL;
static void mix(unsigned char byte) { digest = (digest ^ byte) * 1099511628211ULL; }

template<typename T>
void check_one(const std::vector<T>& input, gef::SplitPointStrategy strategy,
               const gef::BlockParameters<T>* parameters = nullptr) {
    gef::internal::B_STAR_GEF<T> codec(input, strategy, nullptr, parameters);
    if (codec.size() != input.size()) throw std::runtime_error("size");
    for (size_t i = 0; i < input.size(); ++i)
        if (codec[i] != input[i]) throw std::runtime_error("roundtrip");
    const auto path = std::filesystem::temp_directory_path() /
        ("bstar-minmax-differential-" + std::to_string(sizeof(T)) + ".bin");
    codec.serialize(path);
    std::ifstream stream(path, std::ios::binary);
    char byte;
    while (stream.get(byte)) mix(static_cast<unsigned char>(byte));
    stream.close();
    gef::internal::B_STAR_GEF<T> loaded;
    loaded.load(path);
    for (size_t i = 0; i < input.size(); ++i)
        if (loaded[i] != input[i]) throw std::runtime_error("serialized roundtrip");
    std::filesystem::remove(path);
}

template<typename T>
void check(const std::vector<T>& input) {
    check_one(input, gef::APPROXIMATE_SPLIT_POINT);
    check_one(input, gef::OPTIMAL_SPLIT_POINT);
}

template<typename T>
void exercise() {
    std::mt19937_64 rng(125162);
    using U = std::make_unsigned_t<T>;
    const auto random_value = [&]() { return std::bit_cast<T>(static_cast<U>(rng())); };
    constexpr unsigned width = sizeof(T) * 8;
    const T base = std::numeric_limits<T>::min();
    for (unsigned b = 0; b <= width; ++b) {
        const U low_mask = b == width ? U(~U(0)) : U((U(1) << b) - 1);
        std::vector<T> values(131);
        for (size_t i = 0; i < values.size(); ++i) {
            const U high = b == width ? U(0) : U(U((i / 7) & 1) << b);
            const U residual = U(high | (U(rng()) & low_mask));
            values[i] = std::bit_cast<T>(U(U(base) + residual));
        }
        const gef::BlockParameters<T> parameters{base, uint8_t(width), uint8_t(b)};
        check_one(values, gef::APPROXIMATE_SPLIT_POINT, &parameters);
    }
    check<T>({});
    check<T>({std::numeric_limits<T>::min()});
    check<T>({std::numeric_limits<T>::max()});
    check<T>({std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), T(0)});
    for (size_t n : {size_t(2), size_t(3), size_t(7), size_t(31), size_t(32),
                     size_t(33), size_t(63), size_t(64), size_t(65), size_t(1027), size_t(32000)}) {
        std::vector<T> values(n);
        for (auto& value : values) value = random_value();
        check(values);
        std::sort(values.begin(), values.end());
        check(values);
        std::reverse(values.begin(), values.end());
        check(values);
        for (size_t i = 0; i < n; ++i) values[i] = T((i / 17) % 7);
        check(values);
        for (size_t i = 0; i < n; ++i)
            values[i] = i & 1 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        check(values);
        std::fill(values.begin(), values.end(), random_value());
        check(values);
    }
}

int main() {
    exercise<int8_t>(); exercise<uint8_t>();
    exercise<int16_t>(); exercise<uint16_t>();
    exercise<int32_t>(); exercise<uint32_t>();
    exercise<int64_t>(); exercise<uint64_t>();
    std::cout << std::hex << digest << '\n';
}
