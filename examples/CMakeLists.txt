add_executable(mutual_exclusion mutual_exclusion.cpp)
target_link_libraries(mutual_exclusion PRIVATE KripCUDA::kripcuda)
target_compile_options(mutual_exclusion PRIVATE -Wall -Wextra)
