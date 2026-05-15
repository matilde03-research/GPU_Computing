#ifndef MTX_PARSER_H
#define MTX_PARSER_H

#include <vector>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

typedef float FloatType;

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
    int *d_row_ptr;
    int *d_col_idx;
    FloatType *d_values;
};

struct ELLMatrix {
    int m, n, nnz;
    int max_row_len;
    std::vector<int> col_idx;
    std::vector<FloatType> values;
    // GPU pointers
    int *d_col_idx;
    FloatType *d_values;
};

struct JDSMatrix {
    int m, n, nnz;
    std::vector<int> perm;
    std::vector<int> col_start;
    std::vector<int> diag_len;
    std::vector<int> col_idx;
    std::vector<FloatType> values;
    // GPU pointers
    int *d_perm;
    int *d_col_start;
    int *d_diag_len;
    int *d_col_idx;
    FloatType *d_values;
};

// Matrix I/O
COOMatrix readMatrixMarket(const char* filename);
void printMatrixStats(const COOMatrix& coo);

// Conversions with timing
CSRMatrix cooToCSR(const COOMatrix& coo);
ELLMatrix cooToELL(const COOMatrix& coo);
JDSMatrix cooToJDS(const COOMatrix& coo);

// Cleanup
void freeCSRMatrix(CSRMatrix& csr);
void freeELLMatrix(ELLMatrix& ell);
void freeJDSMatrix(JDSMatrix& jds);

// GPU utilities
void generateRandomVector(FloatType *d_x, int n, int seed);

// CPU baseline validation
void csrSpMV_CPU(int m, int n, const int *row_ptr, const int *col_idx, 
                  const FloatType *values, const FloatType *x, FloatType *y);

#endif
