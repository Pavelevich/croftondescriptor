cmake_minimum_required(VERSION 3.18)
project(CroftonCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(crofton_cuda
    main.cu
    border.cpp
)
target_link_libraries(crofton_cuda ${OpenCV_LIBS})
