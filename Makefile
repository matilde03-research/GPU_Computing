NVCC = nvcc
NVCCFLAGS = -arch=sm_75 -std=c++11
INCLUDES = -I./include

# Source and target directories
SRC_DIR = src
BIN_DIR = bin

# CUDA libraries
CUDA_LIBS = -lcuda -lcusparse

# Parser object file
PARSER_OBJ = $(SRC_DIR)/mtx_parser.o

# Main executable targets
TARGETS = $(BIN_DIR)/spmv_csr $(BIN_DIR)/spmv_ell $(BIN_DIR)/spmv_jds $(BIN_DIR)/spmv_cusparse

# Default target
all: $(BIN_DIR) $(PARSER_OBJ) $(TARGETS)

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile mtx_parser.cu to object file
$(PARSER_OBJ): $(SRC_DIR)/mtx_parser.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Build CSR executable
$(BIN_DIR)/spmv_csr: $(SRC_DIR)/spmv_csr.cu $(PARSER_OBJ)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^ $(CUDA_LIBS)

# Build ELL executable
$(BIN_DIR)/spmv_ell: $(SRC_DIR)/spmv_ell.cu $(PARSER_OBJ)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^ $(CUDA_LIBS)

# Build JDS executable
$(BIN_DIR)/spmv_jds: $(SRC_DIR)/spmv_jds.cu $(PARSER_OBJ)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^ $(CUDA_LIBS)

# Build cuSPARSE executable
$(BIN_DIR)/spmv_cusparse: $(SRC_DIR)/spmv_cusparse.cu $(PARSER_OBJ)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^ $(CUDA_LIBS)

# Clean build artifacts
clean:
	rm -f $(PARSER_OBJ) $(TARGETS)

.PHONY: all clean
