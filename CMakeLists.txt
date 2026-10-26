cmake_minimum_required(VERSION 3.20)
project(cuda_kernel_perf_analysis LANGUAGES CXX CUDA)

# ─── C++ / CUDA standards ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── GPU architectures ───────────────────────────────────────────────────────
# sm_75 = Turing (RTX 20xx, Quadro RTX)
# sm_86 = Ampere (RTX 30xx, A-series)
# Add sm_89 for Ada (RTX 40xx) if needed.
set(CMAKE_CUDA_ARCHITECTURES "75;86" CACHE STRING "CUDA architectures to target")

# ─── Compiler flags ──────────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")
endif()

# Enable line-info for Nsight profiling even in Release
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Warn on potential issues
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

# ─── Find CUDA toolkit ───────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ─── Global include directory ────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ─── Subdirectories ──────────────────────────────────────────────────────────
add_subdirectory(src)
add_subdirectory(benchmarks)

# ─── Output results directory (created at configure time) ────────────────────
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/reports)

# ─── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== CUDA Kernel Perf Analysis ===")
message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA toolkit      : ${CUDAToolkit_VERSION}")
message(STATUS "  Install prefix    : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
