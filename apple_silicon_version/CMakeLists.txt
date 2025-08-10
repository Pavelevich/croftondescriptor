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

if(NOT METAL_FRAMEWORK)
    message(FATAL_ERROR "Metal framework not found - required for GPU acceleration")
endif()

if(NOT FOUNDATION_FRAMEWORK)
    message(FATAL_ERROR "Foundation framework not found")
endif()

# Add executables
add_executable(crofton_simple main_simple.cpp)
add_executable(crofton_metal main_metal.cpp crofton_metal.mm)

# Include directories
target_include_directories(crofton_simple PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(crofton_metal PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link libraries for simple version
target_link_libraries(crofton_simple ${OpenCV_LIBS})

# Link libraries for Metal version
target_link_libraries(crofton_metal 
    ${OpenCV_LIBS}
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Compiler-specific options for Apple Silicon optimization
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_compile_options(crofton_simple PRIVATE -mcpu=apple-m1 -O3)
    target_compile_options(crofton_metal PRIVATE -mcpu=apple-m1 -O3)
    message(STATUS "Compiling for Apple Silicon (ARM64)")
else()
    message(STATUS "Compiling for x86_64")
    target_compile_options(crofton_simple PRIVATE -O3 -march=native)
    target_compile_options(crofton_metal PRIVATE -O3 -march=native)
endif()

# Build configuration
set(CMAKE_BUILD_TYPE Release)

# Add custom target for Metal shader compilation (optional - we compile at runtime)
# This is just for IDE support and validation
if(METAL_FRAMEWORK)
    message(STATUS "Metal framework found: ${METAL_FRAMEWORK}")
    message(STATUS "Foundation framework found: ${FOUNDATION_FRAMEWORK}")
    message(STATUS "GPU-accelerated Crofton descriptor available")
endif()