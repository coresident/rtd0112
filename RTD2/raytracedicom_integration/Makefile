# Makefile for RayTraceDicom Integration Project

# Compiler settings
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++14 -O2 -I./include
CXX_FLAGS = -std=c++14 -O2 -I./include
LDFLAGS = -lcudart

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Source files
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
CPP_OBJECTS = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Target executable
TARGET = $(BIN_DIR)/test_raytracedicom

# Default target
all: $(TARGET)

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS) $(CPP_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(OBJECTS) $(CPP_OBJECTS) -o $@ $(LDFLAGS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Install dependencies (CentOS)
install-deps:
	sudo yum update -y
	sudo yum groupinstall -y "Development Tools"
	sudo yum install -y cuda-toolkit

# Test compilation
test: $(TARGET)
	@echo "Running RayTraceDicom integration test..."
	./$(TARGET)

# Show help
help:
	@echo "Available targets:"
	@echo "  all        - Build the complete project"
	@echo "  clean      - Remove build files"
	@echo "  install-deps - Install dependencies (CentOS)"
	@echo "  test       - Build and run the test"
	@echo "  help       - Show this help message"

.PHONY: all clean install-deps test help
