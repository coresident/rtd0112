#!/bin/bash
# Test script for RayTraceDicom Integration Project on CentOS/Linux

echo "=========================================="
echo "RayTraceDicom Integration Project Test"
echo "=========================================="

# Check if we're on CentOS/RHEL
if [ -f /etc/redhat-release ]; then
    echo "✓ Detected CentOS/RHEL system"
    cat /etc/redhat-release
else
    echo "⚠ Warning: Not detected as CentOS/RHEL system"
fi

echo ""
echo "1. Checking system requirements..."

# Check Python
echo "   Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    echo "   ✓ Python $PYTHON_VERSION found"
else
    echo "   ✗ Python3 not found"
fi

# Check CUDA
echo "   Checking CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "   ✓ CUDA $CUDA_VERSION found"
else
    echo "   ✗ CUDA not found"
fi

# Check GCC
echo "   Checking GCC..."
if command -v gcc &> /dev/null; then
    GCC_VERSION=$(gcc --version | head -n1 | sed 's/.*) \([0-9.]*\).*/\1/')
    echo "   ✓ GCC $GCC_VERSION found"
else
    echo "   ✗ GCC not found"
fi

# Check Make
echo "   Checking Make..."
if command -v make &> /dev/null; then
    MAKE_VERSION=$(make --version | head -n1 | sed 's/.*Make \([0-9.]*\).*/\1/')
    echo "   ✓ Make $MAKE_VERSION found"
else
    echo "   ✗ Make not found"
fi

echo ""
echo "2. Checking dependencies..."

# Check pybind11
echo "   Checking pybind11..."
if python3 -c "import pybind11" &> /dev/null; then
    PYBIND11_PATH=$(python3 -c "import pybind11; print(pybind11.get_include())" 2>/dev/null)
    echo "   ✓ pybind11 found at: $PYBIND11_PATH"
else
    echo "   ✗ pybind11 not found"
fi

echo ""
echo "3. Configuration status..."

# Check if config.mk exists
if [ -f config.mk ]; then
    echo "   ✓ config.mk found"
    echo "   Configuration contents:"
    cat config.mk | sed 's/^/     /'
else
    echo "   ✗ config.mk not found"
    echo "   Run './configure.sh' to generate configuration"
fi

echo ""
echo "4. Build status..."

# Check if source files exist
if [ -f test_simplified_wrapper_basic.cpp ]; then
    echo "   ✓ Basic source file found"
else
    echo "   ✗ Basic source file missing"
fi

if [ -f test_simplified_wrapper_optimizesuperposition.cpp ]; then
    echo "   ✓ Full source file found"
else
    echo "   ✗ Full source file missing"
fi

if [ -f Makefile ]; then
    echo "   ✓ Makefile found"
else
    echo "   ✗ Makefile missing"
fi

echo ""
echo "5. Recommendations:"

# Check what needs to be installed
if ! command -v python3 &> /dev/null; then
    echo "   - Install Python3: sudo yum install python3-devel python3-pip"
fi

if ! command -v nvcc &> /dev/null; then
    echo "   - Install CUDA: sudo yum install cuda-toolkit"
fi

if ! command -v gcc &> /dev/null; then
    echo "   - Install development tools: sudo yum groupinstall 'Development Tools'"
fi

if ! python3 -c "import pybind11" &> /dev/null; then
    echo "   - Install pybind11: pip3 install pybind11"
fi

if [ ! -f config.mk ]; then
    echo "   - Run configuration: ./configure.sh"
fi

echo ""
echo "6. Next steps:"
echo "   If all checks pass, you can build the project:"
echo "   - make test-basic    # Build basic version first"
echo "   - make all          # Build both versions"
echo "   - make info         # Show build configuration"

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
