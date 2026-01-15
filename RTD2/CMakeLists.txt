cmake_minimum_required(VERSION 3.15)
project(SimplifiedProtonsWrapper VERSION 1.0.0 LANGUAGES C CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust based on your target GPUs)
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89")

# Compiler-specific flags
if(MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(COMMON_CXX_FLAGS "/MDd /openmp /EHsc /source-charset:utf-8")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(COMMON_CXX_FLAGS "/MD /openmp /EHsc /source-charset:utf-8")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"${COMMON_CXX_FLAGS}\"")
else()
    # For non-MSVC compilers
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3 -lineinfo")
endif()

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Create shared library
add_library(simplified_protons_wrapper SHARED
    simplified_protons_wrapper.cu
)

# Set library properties
set_target_properties(simplified_protons_wrapper PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CXX_VISIBILITY_PRESET "default"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX ""
    SUFFIX ".so"
)

# Link libraries
target_link_libraries(simplified_protons_wrapper PRIVATE cuda cudart)

# Set include directories for the target
target_include_directories(simplified_protons_wrapper PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Create static library version
add_library(simplified_protons_wrapper_static STATIC
    simplified_protons_wrapper.cu
)

set_target_properties(simplified_protons_wrapper_static PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CXX_VISIBILITY_PRESET "default"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

target_link_libraries(simplified_protons_wrapper_static PRIVATE cuda cudart)

target_include_directories(simplified_protons_wrapper_static PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Create test executable
add_executable(test_simplified_wrapper
    test_simplified_wrapper.cpp
)

target_link_libraries(test_simplified_wrapper PRIVATE simplified_protons_wrapper_static)

# Installation
install(TARGETS simplified_protons_wrapper simplified_protons_wrapper_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES simplified_protons_wrapper.cuh
    DESTINATION include
)

# Print configuration information
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

