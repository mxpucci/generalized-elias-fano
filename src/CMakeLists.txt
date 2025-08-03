add_library(gef gef.cpp)
add_library(gef::gef ALIAS gef)

# Set library properties
set_target_properties(gef PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME gef
)

target_include_directories(gef
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link against sdsl publicly so consumers get access to sdsl functionality
# This is appropriate for a library that exposes sdsl types in its interface
target_link_libraries(gef PUBLIC sdsl)

add_executable(ugef ugef.cpp)
target_compile_options(ugef PRIVATE -D_FILE_OFFSET_BITS=64)
target_link_libraries(ugef PRIVATE gef)

add_executable(rle_gef rle_gef.cpp)
target_compile_options(rle_gef PRIVATE -D_FILE_OFFSET_BITS=64)
target_link_libraries(rle_gef PRIVATE gef)

add_executable(bgef bgef.cpp)
target_compile_options(bgef PRIVATE -D_FILE_OFFSET_BITS=64)
target_link_libraries(bgef PRIVATE gef)