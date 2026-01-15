#!/bin/bash
# Simple compilation test script for CentOS/Linux

echo "Testing compilation of test_simplified_wrapper_basic.cpp..."

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

# Try to compile the basic version
echo "Attempting to compile basic version..."
nvcc -std=c++14 -O2 -I. -I./include -I/usr/local/cuda/include -c test_simplified_wrapper_basic.cpp -o test_simplified_wrapper_basic.o

if [ $? -eq 0 ]; then
    echo "✓ Basic version compiled successfully!"
    
    # Try to link
    echo "Attempting to link..."
    nvcc -std=c++14 -O2 -o raytrace_dicom_basic_test test_simplified_wrapper_basic.o -lcudart
    
    if [ $? -eq 0 ]; then
        echo "✓ Basic version linked successfully!"
        echo "✓ Compilation test passed!"
        exit 0
    else
        echo "✗ Linking failed"
        exit 1
    fi
else
    echo "✗ Compilation failed"
    echo "Error details:"
    nvcc -std=c++14 -O2 -I. -I./include -I/usr/local/cuda/include -c test_simplified_wrapper_basic.cpp -o test_simplified_wrapper_basic.o 2>&1
    exit 1
fi
