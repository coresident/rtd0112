#!/bin/bash
# Quick recompile script for RayTraceDicom Integration Project

echo "=========================================="
echo "Quick Recompile Script"
echo "=========================================="

# Check if source files exist
if [ ! -f test_simplified_wrapper_basic.cpp ]; then
    echo "Error: test_simplified_wrapper_basic.cpp not found!"
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
make clean 2>/dev/null || rm -f *.o raytrace_dicom_basic_test raytrace_dicom_test

# Compile basic version
echo "Compiling basic version..."
nvcc -std=c++14 -O2 -I. -I./include -o raytrace_dicom_basic_test test_simplified_wrapper_basic.cpp -lcudart

if [ $? -eq 0 ]; then
    echo "✓ Basic version compiled successfully!"
    
    # Ask if user wants to run the program
    echo ""
    read -p "Do you want to run the program now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running raytrace_dicom_basic_test..."
        echo "=========================================="
        ./raytrace_dicom_basic_test
    else
        echo "To run the program later, use: ./raytrace_dicom_basic_test"
    fi
else
    echo "✗ Compilation failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo "=========================================="
echo "Recompile completed!"
echo "=========================================="
