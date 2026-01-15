#!/bin/bash
# Configuration script to detect Python and pybind11 paths for CentOS/Linux

echo "Detecting Python and pybind11 installation on CentOS/Linux..."

# Detect Python version and path
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "Found Python version: $PYTHON_VERSION"
    PYTHON_PATH=$(python3 -c "import sys; print(sys.prefix + '/include/python' + str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2>/dev/null)
    echo "Python include path: $PYTHON_PATH"
else
    echo "Python3 not found, trying Python..."
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "Found Python version: $PYTHON_VERSION"
        PYTHON_PATH=$(python -c "import sys; print(sys.prefix + '/include/python' + str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2>/dev/null)
        echo "Python include path: $PYTHON_PATH"
    else
        echo "Python not found!"
        echo "Please install Python development packages:"
        echo "  sudo yum install python3-devel python3-pip"
        exit 1
    fi
fi

# Detect pybind11 path
PYBIND11_PATH=$(python3 -c "import pybind11; print(pybind11.get_include())" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "Found pybind11 path: $PYBIND11_PATH"
else
    echo "pybind11 not found via Python, checking common locations..."
    COMMON_PYBIND11_PATHS=(
        "/usr/include/pybind11"
        "/usr/local/include/pybind11"
        "/opt/pybind11/include"
        "/usr/local/lib/python$PYTHON_VERSION/site-packages/pybind11/include"
        "/usr/lib/python$PYTHON_VERSION/site-packages/pybind11/include"
        "/usr/lib64/python$PYTHON_VERSION/site-packages/pybind11/include"
    )
    
    for path in "${COMMON_PYBIND11_PATHS[@]}"; do
        if [ -d "$path" ]; then
            echo "Found pybind11 at: $path"
            PYBIND11_PATH="$path"
            break
        fi
    done
    
    if [ -z "$PYBIND11_PATH" ]; then
        echo "pybind11 not found in common locations!"
        echo "Please install pybind11:"
        echo "  pip3 install pybind11"
        echo "  or"
        echo "  sudo yum install python3-pybind11"
    fi
fi

# Detect CUDA path
CUDA_PATH=$(which nvcc 2>/dev/null | sed 's|/bin/nvcc||')
if [ -n "$CUDA_PATH" ]; then
    echo "Found CUDA at: $CUDA_PATH"
else
    echo "CUDA not found in PATH, checking common locations..."
    COMMON_CUDA_PATHS=(
        "/usr/local/cuda"
        "/opt/cuda"
        "/usr/cuda"
        "/usr/local/cuda-11.0"
        "/usr/local/cuda-11.1"
        "/usr/local/cuda-11.2"
        "/usr/local/cuda-11.3"
        "/usr/local/cuda-11.4"
        "/usr/local/cuda-11.5"
        "/usr/local/cuda-11.6"
        "/usr/local/cuda-11.7"
        "/usr/local/cuda-11.8"
        "/usr/local/cuda-12.0"
        "/usr/local/cuda-12.1"
        "/usr/local/cuda-12.2"
    )
    
    for path in "${COMMON_CUDA_PATHS[@]}"; do
        if [ -d "$path" ]; then
            echo "Found CUDA at: $path"
            CUDA_PATH="$path"
            break
        fi
    done
    
    if [ -z "$CUDA_PATH" ]; then
        echo "CUDA not found in common locations!"
        echo "Please install CUDA Toolkit:"
        echo "  sudo yum install cuda-toolkit"
        echo "  or download from: https://developer.nvidia.com/cuda-downloads"
    fi
fi

# Check for required packages
echo "Checking for required packages..."

# Check for gcc/g++
if ! command -v gcc &> /dev/null; then
    echo "gcc not found! Please install development tools:"
    echo "  sudo yum groupinstall 'Development Tools'"
fi

# Check for make
if ! command -v make &> /dev/null; then
    echo "make not found! Please install development tools:"
    echo "  sudo yum groupinstall 'Development Tools'"
fi

# Generate config.mk
echo "Generating config.mk..."
cat > config.mk << EOF
# Auto-generated configuration for CentOS/Linux
PYTHON_VERSION = $PYTHON_VERSION
PYTHON_PATH = $PYTHON_PATH
PYBIND11_PATH = $PYBIND11_PATH
CUDA_PATH = $CUDA_PATH
EOF

echo "Configuration saved to config.mk"
echo ""
echo "Next steps:"
echo "1. If any packages are missing, install them:"
echo "   make install-deps"
echo "2. Build the project:"
echo "   make test-basic    # Build basic version first"
echo "   make all           # Build both versions"
echo "3. Check configuration:"
echo "   make info"
