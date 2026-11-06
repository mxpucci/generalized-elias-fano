// Standalone baseline/candidate differential driver. Compile against each include
// tree and compare stdout byte-for-byte; a mismatch or roundtrip error fails.
#include "gef/RLE_GEF.hpp"
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
void check(const std::vector<T>& input) {
    gef::internal::RLE_GEF<T> codec(input);
    if (codec.size() != input.size()) throw std::runtime_error("size");
    for (size_t i = 0; i < input.size(); ++i)
        if (codec[i] != input[i]) throw std::runtime_error("roundtrip");
    const auto path = std::filesystem::temp_directory_path() /
        ("rle-minmax-differential-" + std::to_string(sizeof(T)) + ".bin");
    codec.serialize(path);
    std::ifstream stream(path, std::ios::binary);
    char byte;
    while (stream.get(byte)) mix(static_cast<unsigned char>(byte));
    stream.close();
    gef::internal::RLE_GEF<T> loaded;
    loaded.load(path);
    for (size_t i = 0; i < input.size(); ++i)
        if (loaded[i] != input[i]) throw std::runtime_error("serialized roundtrip");
    std::filesystem::remove(path);
}

template<typename T>
void exercise() {
    std::mt19937_64 rng(125162);
    using U = std::make_unsigned_t<T>;
    const auto random_value = [&]() { return std::bit_cast<T>(static_cast<U>(rng())); };
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
