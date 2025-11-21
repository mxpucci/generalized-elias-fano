#include <experimental/simd>
#include <iostream>
namespace stdx = std::experimental;
int main() {
    using V = stdx::native_simd<int>;
    std::cout << V::size() << std::endl;
    return 0;
}
