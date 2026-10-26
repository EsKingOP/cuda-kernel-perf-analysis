##
## src/CMakeLists.txt
##
## Builds the CPU and CUDA implementation libraries that benchmarks link against.
##

# ── CPU library ──────────────────────────────────────────────────────────────
add_library(matmul_cpu STATIC
    cpu/matmul_cpu.cpp
)
target_include_directories(matmul_cpu PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu
)
target_compile_options(matmul_cpu PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /arch:AVX2>
)

# ── CUDA naive library ───────────────────────────────────────────────────────
add_library(matmul_naive_cu STATIC
    cuda/matmul_naive.cu
)
target_include_directories(matmul_naive_cu PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(matmul_naive_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ── CUDA tiled library ───────────────────────────────────────────────────────
add_library(matmul_tiled_cu STATIC
    cuda/matmul_tiled.cu
)
target_include_directories(matmul_tiled_cu PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)
set_target_properties(matmul_tiled_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
