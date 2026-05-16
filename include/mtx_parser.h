#ifndef MTX_PARSER_H
#define MTX_PARSER_H

#include <vector>
#include <string>
#include <cuda_runtime.h>

typedef float FloatType;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct COOMatrix {
    int m, n, nnz;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<FloatType> values;
};

struct CSRMatrix {
    int m, n, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<FloatType> values;
    
    // GPU pointers
    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    FloatType *d_values = nullptr;
    
    // Constructor
    CSRMatrix() = default;
    CSRMatrix(int m_, int n_, int nnz_, 
              std::vector<int> row_ptr_, 
              std::vector<int> col_idx_, 
              std::vector<FloatType> values_)
        : m(m_), n(n_), nnz(nnz_), 
          row_ptr(row_ptr_), col_idx(col_idx_), values(values_) {}
};

struct ELLMatrix {
    int m, n, nnz;
    int max_row_len;
    std::vector<int> col_idx;
    std::vector<FloatType> values;
    
    // GPU pointers
    int *d_col_idx = nullptr;
    FloatType *d_values = nullptr;
    
    // Constructor
    ELLMatrix() = default;
    ELLMatrix(int m_, int n_, int nnz_, int max_row_len_,
              std::vector<int> col_idx_, 
              std::vector<FloatType> values_)
        : m(m_), n(n_), nnz(nnz_), max_row_len(max_row_len_),
          col_idx(col_idx_), values(values_) {}
};


// Function declarations
COOMatrix readMatrixMarket(const char* filename);
CSRMatrix cooToCSR(const COOMatrix& coo);
ELLMatrix cooToELL(const COOMatrix& coo);



void generateRandomVector(FloatType *d_x, int n, int seed);

// Cleanup functions
void freeCSRMatrix(CSRMatrix& csr);
void freeELLMatrix(ELLMatrix& ell);


// CPU baseline validation (OpenMP)
void spmvCPU_CSR(int m, int n, const int *row_ptr, const int *col_idx, 
                 const FloatType *values, const FloatType *x, FloatType *y);

void spmvCPU_ELL(int m, int n, int max_row_len, const int *col_idx, 
                 const FloatType *values, const FloatType *x, FloatType *y);


inline void allocateCSRMatrixGPU(CSRMatrix& csr) {
    CUDA_CHECK(cudaMalloc(&csr.d_row_ptr, (csr.m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&csr.d_col_idx, csr.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&csr.d_values, csr.nnz * sizeof(FloatType)));
    
    CUDA_CHECK(cudaMemcpy(csr.d_row_ptr, csr.row_ptr.data(), (csr.m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(csr.d_col_idx, csr.col_idx.data(), csr.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(csr.d_values, csr.values.data(), csr.nnz * sizeof(FloatType), cudaMemcpyHostToDevice));
}

#endif



