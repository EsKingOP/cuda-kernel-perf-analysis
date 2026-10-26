##
## benchmarks/CMakeLists.txt
##

# ── bench_matmul ─────────────────────────────────────────────────────────────
add_executable(bench_matmul
    bench_matmul.cu
)
target_include_directories(bench_matmul PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_SOURCE_DIR}/src/cuda
)
target_link_libraries(bench_matmul PRIVATE
    matmul_cpu
    matmul_naive_cu
    matmul_tiled_cu
    CUDA::cudart
)
set_target_properties(bench_matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/benchmarks
)

# ── bench_tile_sizes ─────────────────────────────────────────────────────────
add_executable(bench_tile_sizes
    bench_tile_sizes.cu
)
target_include_directories(bench_tile_sizes PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cuda
)
target_link_libraries(bench_tile_sizes PRIVATE
    matmul_tiled_cu
    CUDA::cudart
)
set_target_properties(bench_tile_sizes PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/benchmarks
)
