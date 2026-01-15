cmake_minimum_required(VERSION 3.10)
project(RayTraceDicomIntegration)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14;-O2)

# Include directories
include_directories(include)

# CUDA source files
set(CUDA_SOURCES
    src/ray_tracing.cu
    src/idd_sigma_calculation.cu
    src/superposition_kernels.cu
    src/utils.cu
    src/raytracedicom_wrapper.cu
)

# C++ source files
set(CPP_SOURCES
    src/test_raytracedicom.cpp
)

# Create CUDA library
cuda_add_library(raytracedicom_lib STATIC ${CUDA_SOURCES})

# Create executable
add_executable(test_raytracedicom ${CPP_SOURCES})
target_link_libraries(test_raytracedicom raytracedicom_lib ${CUDA_LIBRARIES})

# Set output directories
set_target_properties(test_raytracedicom PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install targets
install(TARGETS test_raytracedicom
    RUNTIME DESTINATION bin
)

install(TARGETS raytracedicom_lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration info
message(STATUS "CUDA found: ${CUDA_FOUND}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
