#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
#ifdef _OPENMP
    std::cout << "_OPENMP is defined: " << _OPENMP << std::endl;
    std::cout << "omp_get_max_threads(): " << omp_get_max_threads() << std::endl;
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Running with " << omp_get_num_threads() << " threads" << std::endl;
    }
#else
    std::cout << "_OPENMP is NOT defined - OpenMP disabled at compile time" << std::endl;
#endif
    return 0;
}
