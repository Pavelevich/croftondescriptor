cmake_minimum_required(VERSION 3.16)
project(CroftonAppleSilicon LANGUAGES C CXX OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Find required packages
find_package(OpenCV REQUIRED)

# Find Metal and Foundation frameworks (macOS only)
find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(MPS_FRAMEWORK MetalPerformanceShaders)

if(NOT METAL_FRAMEWORK)
    message(FATAL_ERROR "Metal framework not found - required for GPU acceleration")
endif()

if(NOT FOUNDATION_FRAMEWORK)
    message(FATAL_ERROR "Foundation framework not found")
endif()

if(NOT MPS_FRAMEWORK)
    message(FATAL_ERROR "MetalPerformanceShaders framework not found")
endif()

# ============================================================================
# METAL SHADER COMPILATION
# ============================================================================

# Add custom command to compile Metal shaders
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/image_processing.metallib
    COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/image_processing.metal -o ${CMAKE_CURRENT_BINARY_DIR}/image_processing.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/image_processing.air -o ${CMAKE_CURRENT_BINARY_DIR}/image_processing.metallib
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/image_processing.metal
    COMMENT "Compiling Metal shaders to metallib"
    VERBATIM
)

add_custom_target(CompileMetalShaders ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/image_processing.metallib
)

# Copy metallib to build directory for runtime loading
add_custom_command(
    TARGET CompileMetalShaders POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/image_processing.metallib
        ${CMAKE_CURRENT_BINARY_DIR}/image_processing.metallib
    COMMENT "Copying metallib to build directory"
)

# ============================================================================
# EXECUTABLES
# ============================================================================

# Original simple version (CPU only)
add_executable(crofton_simple main_simple.cpp)

# Original Metal version (Crofton descriptor only)
add_executable(crofton_metal main_metal.cpp crofton_metal.mm)

# NEW: Fully optimized Metal version (entire pipeline on GPU)
add_executable(crofton_optimized
    main_metal_optimized.cpp
    MetalImageProcessor.mm
    MetalCapabilities.mm
    crofton_metal.mm
)

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================

target_include_directories(crofton_simple PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(crofton_metal PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(crofton_optimized PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ============================================================================
# LINK LIBRARIES
# ============================================================================

# Simple version (CPU only)
target_link_libraries(crofton_simple ${OpenCV_LIBS})

# Original Metal version
target_link_libraries(crofton_metal
    ${OpenCV_LIBS}
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Optimized Metal version (with MPS)
target_link_libraries(crofton_optimized
    ${OpenCV_LIBS}
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${MPS_FRAMEWORK}
)

# ============================================================================
# COMPILER OPTIMIZATIONS
# ============================================================================

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_compile_options(crofton_simple PRIVATE -mcpu=apple-m1 -O3)
    target_compile_options(crofton_metal PRIVATE -mcpu=apple-m1 -O3)
    target_compile_options(crofton_optimized PRIVATE -mcpu=apple-m1 -O3 -ffast-math)
    message(STATUS "Compiling for Apple Silicon (ARM64)")
else()
    message(STATUS "Compiling for x86_64")
    target_compile_options(crofton_simple PRIVATE -O3 -march=native)
    target_compile_options(crofton_metal PRIVATE -O3 -march=native)
    target_compile_options(crofton_optimized PRIVATE -O3 -march=native -ffast-math)
endif()

# ============================================================================
# BUILD CONFIGURATION
# ============================================================================

set(CMAKE_BUILD_TYPE Release)

# Enable LTO (Link Time Optimization) for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET crofton_optimized PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ============================================================================
# STATUS MESSAGES
# ============================================================================

message(STATUS "")
message(STATUS "╔════════════════════════════════════════════════════════════╗")
message(STATUS "║     Crofton Descriptor - Metal Optimization Build         ║")
message(STATUS "╠════════════════════════════════════════════════════════════╣")
message(STATUS "║ Metal Framework:       ${METAL_FRAMEWORK}")
message(STATUS "║ Foundation Framework:  ${FOUNDATION_FRAMEWORK}")
message(STATUS "║ MPS Framework:         ${MPS_FRAMEWORK}")
message(STATUS "║ OpenCV Version:        ${OpenCV_VERSION}")
message(STATUS "║ Architecture:          ${CMAKE_OSX_ARCHITECTURES}")
message(STATUS "║ Build Type:            ${CMAKE_BUILD_TYPE}")
message(STATUS "╠════════════════════════════════════════════════════════════╣")
message(STATUS "║ Targets:                                                   ║")
message(STATUS "║   - crofton_simple     (CPU baseline)                      ║")
message(STATUS "║   - crofton_metal      (GPU Crofton only)                  ║")
message(STATUS "║   - crofton_optimized  (Full GPU pipeline) ✨              ║")
message(STATUS "╚════════════════════════════════════════════════════════════╝")
message(STATUS "")