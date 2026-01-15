#!/bin/bash
echo "=== RayTraceDicom Integration Compilation ==="

# Create build directories
mkdir -p build bin

echo "Compiling CUDA source files..."

# Compile CUDA source files
nvcc -std=c++14 -O2 -I./include -c src/ray_tracing.cu -o build/ray_tracing.o
if [ $? -ne 0 ]; then
    echo "Error compiling ray_tracing.cu"
    exit 1
fi

nvcc -std=c++14 -O2 -I./include -c src/idd_sigma_calculation.cu -o build/idd_sigma_calculation.o
if [ $? -ne 0 ]; then
    echo "Error compiling idd_sigma_calculation.cu"
    exit 1
fi

nvcc -std=c++14 -O2 -I./include -c src/superposition_kernels.cu -o build/superposition_kernels.o
if [ $? -ne 0 ]; then
    echo "Error compiling superposition_kernels.cu"
    exit 1
fi

nvcc -std=c++14 -O2 -I./include -c src/utils.cu -o build/utils.o
if [ $? -ne 0 ]; then
    echo "Error compiling utils.cu"
    exit 1
fi

nvcc -std=c++14 -O2 -I./include -c src/raytracedicom_wrapper.cu -o build/raytracedicom_wrapper.o
if [ $? -ne 0 ]; then
    echo "Error compiling raytracedicom_wrapper.cu"
    exit 1
fi

echo "Compiling C++ source files..."

# Compile C++ source files
g++ -std=c++14 -O2 -I./include -c src/test_raytracedicom.cpp -o build/test_raytracedicom.o
if [ $? -ne 0 ]; then
    echo "Error compiling test_raytracedicom.cpp"
    exit 1
fi

echo "Linking executable..."

# Link executable
nvcc build/*.o -o bin/test_raytracedicom -lcudart
if [ $? -ne 0 ]; then
    echo "Error linking executable"
    exit 1
fi

echo "Compilation successful!"
echo "Executable created: bin/test_raytracedicom"

echo ""
echo "To run the program:"
echo "  ./bin/test_raytracedicom"
echo ""
echo "To clean build files:"
echo "  rm -rf build/*.o bin/test_raytracedicom"

