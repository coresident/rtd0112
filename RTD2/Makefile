# Makefile for RayTraceDicom Integration Project (CentOS/Linux)
# Handles CUDA, Python, pybind11, and other dependencies

# Include auto-generated configuration if available
-include config.mk

# Compiler settings
NVCC = nvcc
CXX = g++
CC = gcc

# CUDA settings
CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH = -arch=sm_60
CUDA_FLAGS = $(CUDA_ARCH) -std=c++14

# Python settings - adjust these paths according to your system
PYTHON_VERSION ?= 3.8
PYTHON_PATH ?= /usr/include/python$(PYTHON_VERSION)
PYTHON_LIB ?= /usr/lib64/python$(PYTHON_VERSION)

# pybind11 settings - adjust these paths according to your system
PYBIND11_PATH ?= /usr/local/include/pybind11

# CUDA include paths
CUDA_INCLUDES = -I$(CUDA_PATH)/include

# Python and pybind11 includes
PYTHON_INCLUDES = -I$(PYTHON_PATH) -I$(PYBIND11_PATH)

# Additional include paths
INCLUDES = -I. -I./include $(CUDA_INCLUDES) $(PYTHON_INCLUDES)

# Library paths
LIBRARY_PATHS = -L$(CUDA_PATH)/lib64 -L$(PYTHON_LIB)

# Libraries
LIBS = -lcudart -lpython$(PYTHON_VERSION)

# Compiler flags
CXXFLAGS = -std=c++14 -O2 -Wall -fPIC
NVCCFLAGS = $(CUDA_FLAGS) -O2 -Wall

# Source files
CUDA_SOURCES = test_simplified_wrapper_optimizesuperposition.cpp
CUDA_OBJECTS = $(CUDA_SOURCES:.cpp=.o)
BASIC_SOURCES = test_simplified_wrapper_basic.cpp
BASIC_OBJECTS = $(BASIC_SOURCES:.cpp=.o)

# Target executables
TARGET = raytrace_dicom_test
BASIC_TARGET = raytrace_dicom_basic_test

# Default target
all: $(TARGET) $(BASIC_TARGET)

# Main target
$(TARGET): $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(LIBRARY_PATHS) -o $@ $^ $(LIBS)

# Basic target (without pybind11)
$(BASIC_TARGET): $(BASIC_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lcudart

# Compile CUDA/C++ files
%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile .cu files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean target
clean:
	rm -f *.o $(TARGET) $(BASIC_TARGET)

# Install dependencies (CentOS/RHEL)
install-deps:
	sudo yum update -y
	sudo yum groupinstall -y "Development Tools"
	sudo yum install -y python$(PYTHON_VERSION)-devel python$(PYTHON_VERSION)-pip
	sudo yum install -y cuda-toolkit
	pip$(PYTHON_VERSION) install pybind11

# Install dependencies (Ubuntu/Debian)
install-deps-ubuntu:
	sudo apt-get update
	sudo apt-get install -y python$(PYTHON_VERSION)-dev python$(PYTHON_VERSION)-pip
	sudo apt-get install -y nvidia-cuda-toolkit
	pip$(PYTHON_VERSION) install pybind11

# Check Python path
check-python:
	@echo "Checking Python installation..."
	@python$(PYTHON_VERSION) -c "import sys; print('Python path:', sys.path)"
	@python$(PYTHON_VERSION) -c "import pybind11; print('pybind11 path:', pybind11.get_include())"

# Alternative compilation without pybind11 (for testing basic functionality)
test-basic: $(BASIC_TARGET)

# Compile with debug information
debug: NVCCFLAGS += -g -G
debug: CXXFLAGS += -g
debug: $(TARGET) $(BASIC_TARGET)

# Compile with optimization
release: NVCCFLAGS += -O3
release: CXXFLAGS += -O3
release: $(TARGET) $(BASIC_TARGET)

# Show compilation info
info:
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "Python Path: $(PYTHON_PATH)"
	@echo "pybind11 Path: $(PYBIND11_PATH)"
	@echo "Include Paths: $(INCLUDES)"
	@echo "Library Paths: $(LIBRARY_PATHS)"
	@echo "Libraries: $(LIBS)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build both main and basic executables"
	@echo "  $(TARGET)     - Build the main executable with pybind11"
	@echo "  $(BASIC_TARGET) - Build the basic executable without pybind11"
	@echo "  clean        - Remove object files and executables"
	@echo "  install-deps - Install dependencies (CentOS/RHEL)"
	@echo "  install-deps-ubuntu - Install dependencies (Ubuntu/Debian)"
	@echo "  check-python - Check Python and pybind11 installation"
	@echo "  test-basic   - Build basic test without pybind11"
	@echo "  debug        - Build with debug information"
	@echo "  release      - Build with optimization"
	@echo "  info         - Show compilation settings"

.PHONY: all clean install-deps install-deps-ubuntu check-python test-basic debug release info help
