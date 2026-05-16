# GPU_Computing

---

## Supported Formats & Kernels

### **1. COO Format**

#### COO-Standard
#### COO-SortedSegmentedReduction

### **2. CSR Format**

#### CSR-Vector
#### CSR-Flat
#### CSR-Line-Enhance

### **3. ELL Format**
#### ELL-Basic
#### ELL-Coalesced

### **Reference**

#### cuSPARSE

---
```
GPU_Computing/
├── bin/
│    ├── spmv_coo
│    ├── spmv_csr
│    ├── spmv_cusparse
│    ├── spmv_ell
├── include/
│    ├── mtx_parser.h
├── src/
│    ├── mtx_parser.cu
│    ├── mtx_parser.o
│    ├── cspmv_coo.cu
│    ├── cspmv_csr.cu
│    ├── cspmv_cusparse.cu
│    ├── cspmv_ell.cu      
├── ASIC_680ks.mtx   //File to ignore, just for fast checking
├── Makefile
├── README.md
└── batch_script.sh
```

## Building

### Prerequisites

- **CUDA Toolkit** 11.0 or later
- **NVIDIA GPU** with compute capability ≥ 7.5 (tested on A30, RTX-class GPUs)
- **Make** build tool
- **g++** or compatible C++ compiler

### Compilation

```bash
# Build all executables
make

# Run experiments
sbatch batch_script.sh
```
The batch_script.sh file is the one used during class labs and it will generate and output and an error file in a directory called "output"

Important! Before executing the sbatch command is important to check for the last line of the file that is what actually launches the job in the cluster
The line is like that:

./bin/spmv_ell ./data/Rucci1.mtx 4 100 

the first argument is the type of kernels you want to launch, the second is the MatrixMarket file, the third the warmup cycles and the fourth the number of iterations

To download the matrices you can use the download.sh file. It will download the matrices in a directory called "data"
