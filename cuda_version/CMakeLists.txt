cmake_minimum_required(VERSION 3.24)
project(CroftonCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# "native" targets the GPU present at configure time (e.g. sm_120 on RTX 50xx;
# requires CUDA 12.8+ for Blackwell). Override with -DCMAKE_CUDA_ARCHITECTURES.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(crofton_cuda
    main.cu
)
target_link_libraries(crofton_cuda ${OpenCV_LIBS})
