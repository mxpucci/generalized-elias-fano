#pragma once

#include <cstddef>
#include <cstdint>

namespace gef::internal {

// Dynamic storage requested by the two SDSL versions supported here. The
// benchmark's SDSL exposes bit_capacity(); upstream v2 exposes capacity() in
// bits. Both allocate a padding word that serialization does not count. The
// v2 formula describes vectors constructed/loaded at their current size, as
// the codecs do; v2 does not expose retained allocation size after arbitrary
// external resizes within the same final word.
template<class Vector>
size_t sdsl_allocated_payload_bytes(const Vector& vector) {
    if (vector.data() == nullptr) return 0;
    if constexpr (requires { vector.bit_capacity(); }) {
        return vector.bit_capacity() / 8 + sizeof(uint64_t);
    } else {
        return ((vector.bit_size() + 64) / 64) * sizeof(uint64_t);
    }
}

} // namespace gef::internal
