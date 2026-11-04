#ifndef GEF_BLOCK_PARAMETERS_HPP
#define GEF_BLOCK_PARAMETERS_HPP

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace gef {

template<typename T>
struct BlockParameters {
    T base;
    uint8_t width;
    uint8_t split;

    template<typename C>
    void validate(const C& data) const {
        using U = std::make_unsigned_t<T>;
        if (split > width || width > sizeof(T) * 8) {
            throw std::invalid_argument("Invalid GEF block width or split");
        }
        for (const T value : data) {
            const U residual = static_cast<U>(value) - static_cast<U>(base);
            if (value < base || (width < sizeof(T) * 8 && (residual >> width) != 0)) {
                throw std::invalid_argument("Value outside the GEF block universe");
            }
        }
    }
};

}

#endif
