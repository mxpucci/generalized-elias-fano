file(GLOB_RECURSE GEF_SOURCES "*.cpp")

add_library(gef ${GEF_SOURCES})
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