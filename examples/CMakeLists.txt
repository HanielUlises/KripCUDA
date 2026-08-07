add_executable(mutual_exclusion mutual_exclusion.cpp)
target_link_libraries(mutual_exclusion PRIVATE KripCUDA::kripcuda)
target_compile_options(mutual_exclusion PRIVATE -Wall -Wextra)

# Uses the device-facing headers, so it is compiled by NVCC with the same
# dialect handling as the library's own device sources.
unset(CMAKE_CXX_STANDARD)
unset(CMAKE_CXX_STANDARD_REQUIRED)
unset(CMAKE_CXX_EXTENSIONS)
add_executable(compose compose.cu)
target_link_libraries(compose PRIVATE KripCUDA::kripcuda)
target_compile_options(compose PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20 --expt-relaxed-constexpr>)
