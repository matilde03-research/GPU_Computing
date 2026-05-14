#ifndef MTX_PARSER_H
#define MTX_PARSER_H

#include <vector>
#include <string>
#include <cuda_runtime.h>

typedef float FloatType;

// Coordinate format (COO) - for reading from Matrix Market files
struct COOMatrix {
    int m, n;              // Matrix dimensions (m x n)
    int nnz;               // Number of non-zeros
    std::vector<int> row;  // Row indices
    std::vector<int> col;  // Column indices
    std::vector<FloatType> val;  // Values
};

// Compressed Sparse Row (CSR) format
struct CSRMatrix {
    int m, n;              // Matrix dimensions
    int nnz;               // Number of non-zeros
    int *d_row_ptr;        // Device: row pointers (size m+1)
    int *d_col_idx;        // Device: column indices (size nnz)
    FloatType *d_values;   // Device: values (size nnz)
    
    // Host copies for verification
    int *h_row_ptr;
    int *h_col_idx;
    FloatType *h_values;
};

// Ellpack (ELL) format
struct ELLMatrix {
    int m, n;              // Matrix dimensions
    int nnz;               // Total non-zeros
    int max_row_len;       // Maximum non-zeros per row (padded length)
    int *d_col_idx;        // Device: column indices (size m * max_row_len)
    FloatType *d_values;   // Device: values (size m * max_row_len)
    
    int *h_col_idx;
    FloatType *h_values;
};

// Jagged Diagonal Storage (JDS) format
struct JDSMatrix {
    int m, n;              // Matrix dimensions
    int nnz;               // Total non-zeros
    int *d_perm;           // Device: permutation array (row reordering)
    int *d_iperm;          // Device: inverse permutation
    int *d_col_start;      // Device: column start indices
    int *d_diag_len;       // Device: diagonal lengths
    int *d_col_idx;        // Device: column indices (jagged)
    FloatType *d_values;   // Device: values (jagged)
    
    int *h_perm;
    int *h_iperm;
    int *h_col_start;
    int *h_diag_len;
    int *h_col_idx;
    FloatType *h_values;
};

// Function declarations

// Parse Matrix Market file and return COO format
COOMatrix readMatrixMarket(const std::string& filename);

// Convert COO to CSR on device
CSRMatrix cooToCSR(const COOMatrix& coo);
void freeCSRMatrix(CSRMatrix& csr);

// Convert COO to ELL on device
ELLMatrix cooToELL(const COOMatrix& coo);
void freeELLMatrix(ELLMatrix& ell);

// Convert COO to JDS on device
JDSMatrix cooToJDS(const COOMatrix& coo);
void freeJDSMatrix(JDSMatrix& jds);

// Generate random dense vector
void generateRandomVector(FloatType *d_vector, int size, unsigned int seed = 42);

// Print matrix statistics
void printMatrixStats(const COOMatrix& coo);

// CUDA error checking macro
#define CUDA_CHECK(err) \
    do { \
        cudaError_t _err = (err); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif // MTX_PARSER_H
