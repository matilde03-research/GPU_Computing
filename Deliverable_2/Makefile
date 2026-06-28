

NVCC = nvcc


NVCCFLAGS = -arch=sm_80 -std=c++11 -Xcompiler -fopenmp
INCLUDES  = -I./include
# cuSPARSE for the second kernel; MPI linked directly (module provides libmpi).
LIBS      = -lcusparse -lmpi

# Stage the halo exchange through host buffers when MPI is not CUDA-aware.
ifeq ($(CUDA_AWARE),0)
NVCCFLAGS += -DNO_CUDA_AWARE_MPI
endif

# Fallback for a broken/mismatched module assembler: make BINUTILS=/usr/bin
ifdef BINUTILS
NVCCFLAGS += -Xcompiler -B$(BINUTILS)
endif

SRCDIR = src
OBJDIR = obj
BINDIR = bin

PARSER_OBJ = $(OBJDIR)/mtx_parser.o
TARGET     = $(BINDIR)/spmv

all: $(TARGET)

# Support module (host-only): Matrix Market reader + R-MAT generator.
$(PARSER_OBJ): $(SRCDIR)/mtx_parser.cu include/mtx_parser.h
	mkdir -p $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Main combined executable: CSR-vector + cuSPARSE, MPI + OpenMP validation.
$(TARGET): $(SRCDIR)/spmv.cu $(PARSER_OBJ)
	mkdir -p $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SRCDIR)/spmv.cu $(PARSER_OBJ) $(LIBS)

clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
