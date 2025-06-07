#include "gef/gef.hpp"
#include <sdsl/int_vector.hpp>
#include <iostream>

void gef::hello() {
    sdsl::int_vector<> v(10);
    std::cout << "Hello from gef! Created an sdsl int_vector of size " << v.size() << std::endl;
} 