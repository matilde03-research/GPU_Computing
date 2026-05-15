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
};

struct ELLMatrix {
    int m, n, nnz;
    int max_row_len;
    std::vector<int> col_idx;
    std::vector<FloatType> values;
    
    // GPU pointers
    int *d_col_idx = nullptr;
    FloatType *d_values = nullptr;
};

struct JDSMatrix {
    int m, n, nnz;
    std::vector<int> perm;
    std::vector<int> diag_len;
    std::vector<int> col_start;
    std::vector<int> col_idx;
    std::vector<FloatType> values;
    
    // GPU pointers
    int *d_perm = nullptr;
    int *d_diag_len = nullptr;
    int *d_col_start = nullptr;
    int *d_col_idx = nullptr;
    FloatType *d_values = nullptr;
};

// Function declarations
COOMatrix readMatrixMarket(const char* filename);
CSRMatrix cooToCSR(const COOMatrix& coo);
ELLMatrix cooToELL(const COOMatrix& coo);
JDSMatrix cooToJDS(const COOMatrix& coo);

void printMatrixStats(const COOMatrix& coo);
void generateRandomVector(FloatType *d_x, int n, int seed);

// Cleanup functions
void freeCSRMatrix(CSRMatrix& csr);
void freeELLMatrix(ELLMatrix& ell);
void freeJDSMatrix(JDSMatrix& jds);

// CPU baseline validation (OpenMP)
void spmvCPU_CSR(int m, int n, const int *row_ptr, const int *col_idx, 
                 const FloatType *values, const FloatType *x, FloatType *y);

void spmvCPU_ELL(int m, int n, int max_row_len, const int *col_idx, 
                 const FloatType *values, const FloatType *x, FloatType *y);

void spmvCPU_JDS(int m, int n, const int *perm, const int *diag_len, const int *col_start,
                 const int *col_idx, const FloatType *values, const FloatType *x, FloatType *y);

#endif
