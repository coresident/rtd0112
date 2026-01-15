# Makefile for RTD Integration Project

# Compiler settings
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -std=c++14 -O2 -I./include --compiler-options -fPIC -arch=sm_86
CXX_FLAGS = -std=c++14 -O2 -I./include -fPIC
LDFLAGS = -lcudart

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Source files
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
# Exclude test files and debug main from main library
LIBRARY_CUDA_SOURCES = $(filter-out $(SRC_DIR)/test_%.cu $(SRC_DIR)/simple_timing_demo.cu $(SRC_DIR)/debug_main.cu $(SRC_DIR)/simple_debug.cu $(SRC_DIR)/cuda_diagnostic.cu $(SRC_DIR)/complete_dose_debug.cu $(SRC_DIR)/dose_debug_detailed.cu $(SRC_DIR)/simple_dose_test.cu $(SRC_DIR)/table_loader.cu, $(CUDA_SOURCES))
OBJECTS = $(LIBRARY_CUDA_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
CPP_OBJECTS = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Target shared library
TARGET = $(BIN_DIR)/libraytracedicom.so

# Debug executable
DEBUG_TARGET = $(BIN_DIR)/debug_raytracedicom
SIMPLE_DEBUG_TARGET = $(BIN_DIR)/simple_debug
COMPLETE_DEBUG_TARGET = $(BIN_DIR)/complete_dose_debug
DETAILED_DEBUG_TARGET = $(BIN_DIR)/dose_debug_detailed
SIMPLE_DOSE_TARGET = $(BIN_DIR)/simple_dose_test
TABLE_LOADER_TARGET = $(BIN_DIR)/table_loader

# Default target
all: $(TARGET) $(DEBUG_TARGET) $(SIMPLE_DEBUG_TARGET) $(COMPLETE_DEBUG_TARGET) $(DETAILED_DEBUG_TARGET) $(SIMPLE_DOSE_TARGET) $(TABLE_LOADER_TARGET)

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

# Link shared library
$(TARGET): $(OBJECTS) $(CPP_OBJECTS) | $(BIN_DIR)
	$(NVCC) -shared $(OBJECTS) $(CPP_OBJECTS) -o $@ $(LDFLAGS)

# Compile debug executable
$(DEBUG_TARGET): $(SRC_DIR)/debug_main.cu $(TARGET) | $(BIN_DIR)
	$(NVCC) -std=c++14 -O2 -I./include -arch=sm_86 $(SRC_DIR)/debug_main.cu -o $@ -L$(BIN_DIR) -lraytracedicom $(LDFLAGS)

# Compile simple debug executable
$(SIMPLE_DEBUG_TARGET): $(SRC_DIR)/simple_debug.cu | $(BIN_DIR)
	$(NVCC) -std=c++14 -O2 -I./include $(SRC_DIR)/simple_debug.cu -o $@ $(LDFLAGS)

# Compile complete dose debug executable
$(COMPLETE_DEBUG_TARGET): $(SRC_DIR)/complete_dose_debug.cu $(TARGET) | $(BIN_DIR)
	$(NVCC) -std=c++14 -O2 -I./include -arch=sm_86 $(SRC_DIR)/complete_dose_debug.cu -o $@ -L$(BIN_DIR) -lraytracedicom $(LDFLAGS)

# Compile detailed dose debug executable
$(DETAILED_DEBUG_TARGET): $(SRC_DIR)/dose_debug_detailed.cu $(TARGET) | $(BIN_DIR)
	$(NVCC) -std=c++14 -O2 -I./include -arch=sm_86 $(SRC_DIR)/dose_debug_detailed.cu -o $@ -L$(BIN_DIR) -lraytracedicom $(LDFLAGS)

# Compile simple dose test executable
$(SIMPLE_DOSE_TARGET): $(SRC_DIR)/simple_dose_test.cu $(TARGET) | $(BIN_DIR)
	$(NVCC) -std=c++14 -O2 -I./include -arch=sm_86 $(SRC_DIR)/simple_dose_test.cu -o $@ -L$(BIN_DIR) -lraytracedicom $(LDFLAGS)

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
	@echo "Running RTD integration test..."
	./$(TARGET)

# Run debug program
debug: $(DEBUG_TARGET)
	@echo "Running debug program..."
	LD_LIBRARY_PATH=$(BIN_DIR):/usr/local/cuda/targets/x86_64-linux/lib ./$(DEBUG_TARGET)

# Run simple debug program
simple-debug: $(SIMPLE_DEBUG_TARGET)
	@echo "Running simple debug program..."
	./$(SIMPLE_DEBUG_TARGET)

# Run complete dose debug program
complete-debug: $(COMPLETE_DEBUG_TARGET)
	@echo "Running complete dose debug program..."
	LD_LIBRARY_PATH=$(BIN_DIR):/usr/local/cuda/targets/x86_64-linux/lib ./$(COMPLETE_DEBUG_TARGET)

# Run detailed dose debug program
detailed-debug: $(DETAILED_DEBUG_TARGET)
	@echo "Running detailed dose debug program..."
	LD_LIBRARY_PATH=$(BIN_DIR):/usr/local/cuda/targets/x86_64-linux/lib ./$(DETAILED_DEBUG_TARGET)

# Run simple dose test program
simple-dose-test: $(SIMPLE_DOSE_TARGET)
	@echo "Running simple dose test program..."
	LD_LIBRARY_PATH=$(BIN_DIR):/usr/local/cuda/targets/x86_64-linux/lib ./$(SIMPLE_DOSE_TARGET)

# Show help
help:
	@echo "Available targets:"
	@echo "  all        - Build the complete project (library + debug executable)"
	@echo "  clean      - Remove build files"
	@echo "  install-deps - Install dependencies (CentOS)"
	@echo "  test       - Build and run the test"
	@echo "  debug      - Build and run the debug program"
	@echo "  simple-debug - Build and run the simple debug program"
	@echo "  complete-debug - Build and run the complete dose calculation debug program"
	@echo "  detailed-debug - Build and run the detailed dose debugging program"
	@echo "  simple-dose-test - Build and run the simple dose calculation test"
	@echo "  help       - Show this help message"

.PHONY: all clean install-deps test debug simple-debug complete-debug detailed-debug simple-dose-test help
