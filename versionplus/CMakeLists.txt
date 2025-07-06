cmake_minimum_required(VERSION 3.30)
project(untitled LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(untitled main.cu
        border.cpp)
target_link_libraries(untitled ${OpenCV_LIBS})
