cmake_minimum_required(VERSION 3.18)
project(secondary_gpu_cache LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Include directories
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA library dirs: ${CUDAToolkit_LIBRARY_DIRS}")

# Define the shared library
add_library(secondary_gpu_cache SHARED
    src/secondary_gpu_cache.cpp
    src/bindings.cpp
)

# Include dirs and libs for target
target_include_directories(secondary_gpu_cache PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(secondary_gpu_cache
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

# Remove the 'lib' prefix
set_target_properties(secondary_gpu_cache PROPERTIES PREFIX "")

# Let PyTorch handle CUDA arch
set_property(TARGET secondary_gpu_cache PROPERTY CUDA_ARCHITECTURES OFF)
