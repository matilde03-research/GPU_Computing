cmake_minimum_required(VERSION 3.20)
project(SpMV_GPU CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set CUDA architecture (adjust based on your GPU)
# For modern GPUs:
# - RTX 20/30/40 series: sm_75 or higher
# - A100/H100: sm_80 or sm_90
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

# Add optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Create matrix parser library
add_library(mtx_parser OBJECT
    src/mtx_parser.cu
)
set_target_properties(mtx_parser PROPERTIES
    LANGUAGE CUDA
)

# CSR benchmark executable
add_executable(spmv_csr
    src/spmv_csr.cu
    $<TARGET_OBJECTS:mtx_parser>
)
set_target_properties(spmv_csr PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(spmv_csr ${CUDA_LIBRARIES})

# ELL benchmark executable
add_executable(spmv_ell
    src/spmv_ell.cu
    $<TARGET_OBJECTS:mtx_parser>
)
set_target_properties(spmv_ell PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(spmv_ell ${CUDA_LIBRARIES})

# JDS benchmark executable
add_executable(spmv_jds
    src/spmv_jds.cu
    $<TARGET_OBJECTS:mtx_parser>
)
set_target_properties(spmv_jds PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(spmv_jds ${CUDA_LIBRARIES})

# Set output directory
set_target_properties(spmv_csr spmv_ell spmv_jds PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Print build information
message(STATUS "CUDA Architectures: sm_75 (adjust if needed)")
message(STATUS "Data type: Float32 (single precision)")
message(STATUS "Block size: 256 threads")
message(STATUS "Random seed: 42 (fixed for reproducibility)")
