function(kripcuda_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE KripCUDA::kripcuda)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(${name} PRIVATE -Wall -Wextra)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Tests that use the device-facing headers are compiled by NVCC, and need the
# same dialect handling as the library's own device sources.
function(kripcuda_add_cuda_test name)
    unset(CMAKE_CXX_STANDARD)
    unset(CMAKE_CXX_STANDARD_REQUIRED)
    unset(CMAKE_CXX_EXTENSIONS)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} PRIVATE KripCUDA::kripcuda)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20 --expt-relaxed-constexpr>)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

kripcuda_add_test(test_kripke)
kripcuda_add_test(test_state_set)
kripcuda_add_test(test_reachability)
kripcuda_add_test(test_ctl)
kripcuda_add_cuda_test(test_device)
kripcuda_add_cuda_test(test_scc)
