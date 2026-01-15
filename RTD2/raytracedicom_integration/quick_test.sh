#!/bin/bash
echo "=== RayTraceDicom Integration Quick Test ==="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA not found. Please install CUDA Toolkit."
    echo "For CentOS: sudo yum install cuda-toolkit"
    echo "For Ubuntu: sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not found or driver not installed."
    exit 1
fi

# Display GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Check if we're in the right directory
if [ ! -f "compile_all.sh" ]; then
    echo "Error: compile_all.sh not found. Please run this script from the raytracedicom_integration directory."
    exit 1
fi

# Compile
echo "Compiling RayTraceDicom Integration..."
chmod +x compile_all.sh
./compile_all.sh

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    
    # Run test
    echo "Running RayTraceDicom test..."
    ./bin/test_raytracedicom
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Test completed successfully!"
        echo "RayTraceDicom Integration is working correctly."
    else
        echo ""
        echo "❌ Test failed!"
        echo "Please check the error messages above."
        exit 1
    fi
else
    echo ""
    echo "❌ Compilation failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "=== Test Complete ==="

