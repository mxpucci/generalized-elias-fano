# Target system
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Cross compiler
set(CMAKE_C_COMPILER /opt/homebrew/bin/x86_64-linux-musl-gcc)
set(CMAKE_CXX_COMPILER /opt/homebrew/bin/x86_64-linux-musl-g++)

# Disable macOS-specific flags early
set(CMAKE_OSX_SYSROOT "")
set(CMAKE_OSX_ARCHITECTURES "")

# Tell CMake to NOT test macOS flags
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Ensure static linking
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")

# Optimizations
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -march=skylake-avx512 -mtune=skylake-avx512 -flto")
