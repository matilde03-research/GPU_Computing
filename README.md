# SpMV GPU Computing - Sparse Matrix Vector Multiplication Benchmark

This project implements efficient SpMV (Sparse Matrix-Vector Multiplication) operations on GPUs using CUDA in C. It provides three matrix storage formats with multiple kernel implementations for performance comparison.

## Project Structure

```
.
├── CMakeLists.txt              # Build configuration
├── include/
│   └── mtx_parser.h            # Matrix Market parser and data structures
├── src/
│   ├── mtx_parser.cu           # Matrix Market file parser implementation
│   ├── spmv_csr.cu             # CSR format with 3 kernels
│   ├── spmv_ell.cu             # ELL format with 2 kernels
│   └── spmv_jds.cu             # JDS format with 2 kernels
├── CMakeLists.txt              # Build configuration
└── README.md                    # This file
```

## Features

### 1. **CSR (Compressed Sparse Row) Format**
Three optimized kernels for SpMV computation:
- **CSR-Vector**: One thread per row (baseline)
- **CSR-Flat**: Multiple threads per row using warp-level parallelism
- **CSR-Line-Enhance**: Adaptive parallelization with advanced load balancing (Chu et al. method)

### 2. **ELL (Ellpack) Format**
Two kernels with different optimization strategies:
- **ELL-Basic**: Straightforward one-thread-per-row parallelization
- **ELL-Optimized**: Shared memory optimization and improved load balancing

### 3. **JDS (Jagged Diagonal Storage) Format**
Two kernels optimized for irregular sparsity patterns:
- **JDS-Basic**: Direct row-sorted parallelization
- **JDS-Optimized**: Warp-level parallelism with shared memory optimization

### Data Types
- All computations use **Float32** precision
- Random vectors generated with fixed seed (42) for reproducibility

### Performance Metrics
- Execution time (milliseconds)
- GFLOP/s (Gigaflops per second)
- Automatic calculation based on: 2 × NNZ floating point operations per SpMV

## Building the Project

### Prerequisites
- CUDA Toolkit (11.0 or later)
- CMake 3.20+
- NVIDIA GPU with compute capability 7.5 or higher

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make -j4
```

This will generate three executables:
- `spmv_csr` - CSR format benchmark
- `spmv_ell` - ELL format benchmark
- `spmv_jds` - JDS format benchmark

## Usage

### Running the Benchmarks

Each executable follows the same command-line interface:

```bash
./spmv_csr <matrix.mtx> <warmup_cycles> <iterations>
./spmv_ell <matrix.mtx> <warmup_cycles> <iterations>
./spmv_jds <matrix.mtx> <warmup_cycles> <iterations>
```

### Parameters
- `<matrix.mtx>`: Path to a Matrix Market format file
- `<warmup_cycles>`: Number of warm-up runs (integer > 0)
- `<iterations>`: Number of timed iterations (integer > 0)

### Example

```bash
./spmv_csr matrix.mtx 10 100
./spmv_ell matrix.mtx 10 100
./spmv_jds matrix.mtx 10 100
```

### Output Example

```
╔════════════════════════════════════════════════════════════╗
║           CSR Format SpMV Benchmark                       ║
╚════════════════════════════════════════════════════════════╝

Input file: matrix.mtx
Warmup cycles: 10
Iterations: 100

Matrix Market file loaded successfully
Dimensions: 1000 x 1000
Non-zeros: 50000
Sparsity: 5.00%
Memory (CSR): 204 KB

=== CSR SpMV Benchmark ===
Matrix: 1000 x 1000, NNZ: 50000
Grid size: 4, Block size: 256

Kernel 1: CSR-Vector (one thread per row)
  Average time: 0.1234 ms
  GFLOP/s: 812.36

Kernel 2: CSR-Flat (warp per row)
  Average time: 0.0987 ms
  GFLOP/s: 1013.21

Kernel 3: CSR-Line-Enhance (adaptive parallelism)
  Average time: 0.0945 ms
  GFLOP/s: 1058.52

✓ CSR benchmark completed successfully!
```

## Matrix Market Format

The program expects input matrices in Matrix Market coordinate format (.mtx files):

```
%%MatrixMarket matrix coordinate real general
% Comment lines starting with %
rows cols nnz
row1 col1 value1
row2 col2 value2
...
```

Example matrices can be found at: https://math.nist.gov/MatrixMarket/

## Implementation Details

### Matrix Format Conversions

1. **COO to CSR**: Row-wise compression with row pointers for efficient row access
2. **COO to ELL**: Padding to equal length rows for aligned access patterns
3. **COO to JDS**: Jagged diagonal with row reordering for improved load balance

### GPU Optimization Techniques

- **Coalesced Memory Access**: Sequential data layout for 32-byte transactions
- **Shared Memory**: On-chip caching to reduce global memory pressure
- **Warp-level Reductions**: Efficient __shfl_down_sync for parallel sums
- **Load Balancing**: Dynamic work distribution based on row density
- **Bank Conflict Avoidance**: Strategic padding in shared memory access

### Performance Calculation

GFLOP/s = (2 × NNZ) / (avg_time_ms × 1e-3) / 1e9

Where NNZ is the total number of non-zeros in the matrix.

## References

The CSR Flat and Line-Enhance methods are based on:
- **"Efficient Algorithm Design of Optimizing SpMV on GPU"** 
  - Chu, G., He, Y., Dong, L., Ding, Z., Chen, D., Bai, H., Wang, X., Hu, C.

## Implementation Notes

- Random vector seed: 42 (fixed for reproducibility)
- Data type: Float32 (single precision)
- Block size: 256 threads per block (tunable in source)
- Results are not verified (focus is on performance measurement)
- Warm-up cycles eliminate GPU clock ramping effects
- Multiple iterations provide stable averaged timings

## Future Enhancements

- Adaptive kernel selection based on matrix properties
- Additional formats: BSR, BCSR, COO
- Multi-GPU support with domain decomposition
- Auto-tuning system for parameter optimization
- Result verification against reference implementations
- Support for double precision (Float64)

---

**Created for:** GPU Computing Research  
**Repository:** matilde03-research/GPU_Computing
