# Makefile for GPU_Computing project
# Compiles CUDA and C++ source files to executables

# Compiler definitions
NVCC := nvcc
CXX := g++
CXXFLAGS := -Wall -Wextra -O2
NVCCFLAGS := -arch=sm_75 -O2

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# Find all source files
CUDA_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
ALL_SOURCES := $(CUDA_SOURCES) $(CPP_SOURCES)

# Generate executable names from source files
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(CUDA_SOURCES)) \
               $(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/%,$(CPP_SOURCES))

# Default target
.PHONY: all clean

all: $(BIN_DIR) $(EXECUTABLES)

# Create bin directory
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Compile CUDA files to executables
$(BIN_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Compile C++ files to executables
$(BIN_DIR)/%: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Clean up executables
clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
