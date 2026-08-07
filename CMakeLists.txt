cmake_minimum_required(VERSION 3.22)

project(KripCUDA
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated explicit-state model checking of Kripke structures"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(KRIPCUDA_CUDA_ARCHITECTURES "75;86;89"
    CACHE STRING "CUDA architectures to generate code for")
set(CMAKE_CUDA_ARCHITECTURES "${KRIPCUDA_CUDA_ARCHITECTURES}")

option(KRIPCUDA_BUILD_TESTS "Build the KripCUDA test suite" ON)
option(KRIPCUDA_BUILD_EXAMPLES "Build the KripCUDA examples" ON)

find_package(CUDAToolkit REQUIRED)

add_library(kripcuda_headers INTERFACE)
target_include_directories(kripcuda_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Device sources live in their own object library. CMake 3.22 cannot express the
# CUDA20 dialect for NVCC, yet derives it from CMAKE_CXX_STANDARD and then fails
# to generate; clearing that variable for this target only lets us request
# -std=c++20 from NVCC directly, which the public headers need (std::span).
unset(CMAKE_CXX_STANDARD)
unset(CMAKE_CXX_STANDARD_REQUIRED)
unset(CMAKE_CXX_EXTENSIONS)
add_library(kripcuda_device OBJECT
    src/cuda/device_kripke.cu
    src/cuda/scan.cu
    src/cuda/state_set_ops.cu
    src/cuda/reachability_device.cu
    src/cuda/ctl_device.cu
    src/cuda/product.cu
    src/cuda/transpose.cu
    src/cuda/scc_device.cu
    src/cuda/fairness_device.cu)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

target_link_libraries(kripcuda_device PUBLIC kripcuda_headers)
target_compile_options(kripcuda_device PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20 --expt-relaxed-constexpr>)
set_target_properties(kripcuda_device PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(kripcuda
    src/model/kripke.cpp
    src/model/state_set.cpp
    src/exploration/reachability_host.cpp
    src/verification/formula.cpp
    src/verification/ctl_host.cpp
    $<TARGET_OBJECTS:kripcuda_device>)

target_link_libraries(kripcuda PUBLIC kripcuda_headers CUDA::cudart)
target_compile_options(kripcuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
set_target_properties(kripcuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(KripCUDA::kripcuda ALIAS kripcuda)

if(KRIPCUDA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(KRIPCUDA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
