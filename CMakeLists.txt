cmake_minimum_required(VERSION 3.10)
project(RayTraceDicomIntegration)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Set CUDA properties
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY CUDA_ARCHITECTURES 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/core)
include_directories(${CMAKE_SOURCE_DIR}/include/algorithms)
include_directories(${CMAKE_SOURCE_DIR}/include/utils)

# Core source files
set(CORE_SOURCES
    src/core/raytracedicom_wrapper.cu
    src/core/ray_tracing.cu
    src/core/bev_ray_tracing.cu
)

# Algorithm source files
set(ALGORITHM_SOURCES
    src/algorithms/convolution.cu
    src/algorithms/superposition.cu
    src/algorithms/complete_superposition.cu
    src/algorithms/idd_sigma.cu
    src/algorithms/cuda_final_dose.cu
    src/algorithms/prim_transf_kernel.cu
    src/algorithms/transfer_param_struct_div3_impl.cu
    src/algorithms/transfer_param_helper.cu
)

# Utility source files
set(UTILITY_SOURCES
    src/utils/utils.cu
    src/utils/texture_ultra_optimized.cu
    src/utils/advanced_memory_pool.cu
    src/utils/precompiled_texture_manager.cu
    src/utils/energy_reader.cpp
    src/utils/backward_compatibility.cu
)

# Test source files
set(TEST_SOURCES
    src/tests/complete_dose_debug.cu
)

# All source files
set(ALL_SOURCES
    ${CORE_SOURCES}
    ${ALGORITHM_SOURCES}
    ${UTILITY_SOURCES}
)

# Create main library
add_library(raytracedicom_lib ${ALL_SOURCES})

# Set CUDA properties for the library
set_property(TARGET raytracedicom_lib PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86)
set_property(TARGET raytracedicom_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA libraries
target_link_libraries(raytracedicom_lib ${CUDA_LIBRARIES})

# Create test executable
add_executable(raytracedicom_test ${TEST_SOURCES})
target_link_libraries(raytracedicom_test raytracedicom_lib)

# Create CPB debug executable
add_executable(cpb_debug_test src/tests/cpb_convolution_debug.cu)
target_link_libraries(cpb_debug_test raytracedicom_lib)

# Create texture coordinate test executable
add_executable(texture_coordinate_test src/tests/texture_coordinate_test.cu)
target_link_libraries(texture_coordinate_test raytracedicom_lib)

add_executable(simple_texture_debug src/tests/simple_texture_debug.cu)
target_link_libraries(simple_texture_debug raytracedicom_lib)

# Add CPB to Ray Tracing Integration Test
add_executable(cpb_ray_tracing_integration src/tests/cpb_ray_tracing_integration.cu)
target_link_libraries(cpb_ray_tracing_integration raytracedicom_lib)

# Wrapper integration test
add_executable(wrapper_integration_test src/tests/wrapper_integration_test.cu)
target_link_libraries(wrapper_integration_test raytracedicom_lib)

# Complete dose debug test
add_executable(complete_dose_debug src/tests/complete_dose_debug.cu)
target_link_libraries(complete_dose_debug raytracedicom_lib)

# Simple ray mapping test
add_executable(simple_ray_mapping_test src/tests/simple_ray_mapping_test.cu)
target_link_libraries(simple_ray_mapping_test raytracedicom_lib)

# Dose distribution analysis test
add_executable(dose_distribution_analysis src/tests/dose_distribution_analysis.cu)
target_link_libraries(dose_distribution_analysis raytracedicom_lib)

# Set output directories and CUDA architectures
set_target_properties(raytracedicom_lib raytracedicom_test cpb_debug_test texture_coordinate_test simple_texture_debug cpb_ray_tracing_integration wrapper_integration_test complete_dose_debug simple_ray_mapping_test dose_distribution_analysis
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CUDA_ARCHITECTURES 86
)

# Compiler-specific options
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(raytracedicom_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        -Xcompiler -Wall
        >
    )
endif()

# Print configuration summary
message(STATUS "RayTraceDicom Integration Configuration:")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "  Core Sources: ${CORE_SOURCES}")
message(STATUS "  Algorithm Sources: ${ALGORITHM_SOURCES}")
message(STATUS "  Utility Sources: ${UTILITY_SOURCES}")
message(STATUS "  Test Sources: ${TEST_SOURCES}")
