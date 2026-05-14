#include "../include/mtx_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <map>
#include <curand_kernel.h>

// Parse Matrix Market file
COOMatrix readMatrixMarket(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }

    COOMatrix coo = {0, 0, 0, {}, {}, {}};
    std::string line;
    
    // Read header
    while (std::getline(file, line)) {
        if (line[0] != '%') break;  // Skip comment lines
    }

    // Parse dimensions and nnz
    std::istringstream iss(line);
    if (!(iss >> coo.m >> coo.n >> coo.nnz)) {
        fprintf(stderr, "Error: Invalid Matrix Market header\n");
        exit(EXIT_FAILURE);
    }

    coo.row.reserve(coo.nnz);
    coo.col.reserve(coo.nnz);
    coo.val.reserve(coo.nnz);

    // Read matrix entries (1-indexed in file, convert to 0-indexed)
    int row_idx, col_idx;
    FloatType value;
    while (std::getline(file, line)) {
        std::istringstream entry(line);
        if (entry >> row_idx >> col_idx >> value) {
            coo.row.push_back(row_idx - 1);  // Convert to 0-indexed
            coo.col.push_back(col_idx - 1);
            coo.val.push_back(value);
        }
    }

    file.close();
    
    if ((int)coo.row.size() != coo.nnz) {
        fprintf(stderr, "Warning: Read %lu entries but header specified %d\n", 
                coo.row.size(), coo.nnz);
        coo.nnz = (int)coo.row.size();
    }

    return coo;
}

// Convert COO to CSR format
CSRMatrix cooToCSR(const COOMatrix& coo) {
    CSRMatrix csr;
    csr.m = coo.m;
    csr.n = coo.n;
    csr.nnz = coo.nnz;

    // Create host CSR data
    csr.h_row_ptr = new int[coo.m + 1]();
    csr.h_col_idx = new int[coo.nnz];
    csr.h_values = new FloatType[coo.nnz];

    // Count non-zeros per row
    for (int i = 0; i < coo.nnz; i++) {
        csr.h_row_ptr[coo.row[i] + 1]++;
    }

    // Compute row pointers
    for (int i = 0; i < coo.m; i++) {
        csr.h_row_ptr[i + 1] += csr.h_row_ptr[i];
    }

    // Sort entries by row and fill CSR
    std::vector<int> row_count(coo.m, 0);
    for (int i = 0; i < coo.nnz; i++) {
        int row = coo.row[i];
        int pos = csr.h_row_ptr[row] + row_count[row]++;
        csr.h_col_idx[pos] = coo.col[i];
        csr.h_values[pos] = coo.val[i];
    }

    // Copy to device
    CUDA_CHECK(cudaMalloc(&csr.d_row_ptr, (coo.m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&csr.d_col_idx, coo.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&csr.d_values, coo.nnz * sizeof(FloatType)));

    CUDA_CHECK(cudaMemcpy(csr.d_row_ptr, csr.h_row_ptr, (coo.m + 1) * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(csr.d_col_idx, csr.h_col_idx, coo.nnz * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(csr.d_values, csr.h_values, coo.nnz * sizeof(FloatType), 
                          cudaMemcpyHostToDevice));

    return csr;
}

// Convert COO to ELL format
ELLMatrix cooToELL(const COOMatrix& coo) {
    ELLMatrix ell;
    ell.m = coo.m;
    ell.n = coo.n;
    ell.nnz = coo.nnz;

    // Find maximum row length
    std::vector<int> row_lengths(coo.m, 0);
    for (int i = 0; i < coo.nnz; i++) {
        row_lengths[coo.row[i]]++;
    }
    ell.max_row_len = *std::max_element(row_lengths.begin(), row_lengths.end());

    // Create host ELL data (padded to max_row_len)
    ell.h_col_idx = new int[coo.m * ell.max_row_len];
    ell.h_values = new FloatType[coo.m * ell.max_row_len];
    
    // Initialize with invalid column index (-1) and zero values
    for (int i = 0; i < coo.m * ell.max_row_len; i++) {
        ell.h_col_idx[i] = -1;
        ell.h_values[i] = 0.0f;
    }

    // Fill ELL format (column-major order for better memory access)
    std::vector<int> col_count(coo.m, 0);
    for (int i = 0; i < coo.nnz; i++) {
        int row = coo.row[i];
        int col_pos = col_count[row]++;
        ell.h_col_idx[col_pos * coo.m + row] = coo.col[i];
        ell.h_values[col_pos * coo.m + row] = coo.val[i];
    }

    // Copy to device
    CUDA_CHECK(cudaMalloc(&ell.d_col_idx, coo.m * ell.max_row_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ell.d_values, coo.m * ell.max_row_len * sizeof(FloatType)));

    CUDA_CHECK(cudaMemcpy(ell.d_col_idx, ell.h_col_idx, coo.m * ell.max_row_len * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ell.d_values, ell.h_values, coo.m * ell.max_row_len * sizeof(FloatType), 
                          cudaMemcpyHostToDevice));

    return ell;
}

// Convert COO to JDS format
JDSMatrix cooToJDS(const COOMatrix& coo) {
    JDSMatrix jds;
    jds.m = coo.m;
    jds.n = coo.n;
    jds.nnz = coo.nnz;

    // Count non-zeros per row
    std::vector<int> row_lengths(coo.m, 0);
    for (int i = 0; i < coo.nnz; i++) {
        row_lengths[coo.row[i]]++;
    }

    // Create permutation based on row lengths (descending)
    std::vector<std::pair<int, int>> row_pairs;
    for (int i = 0; i < coo.m; i++) {
        row_pairs.push_back({row_lengths[i], i});
    }
    std::sort(row_pairs.begin(), row_pairs.end(), std::greater<std::pair<int, int>>());

    // Create permutation and inverse permutation arrays
    jds.h_perm = new int[coo.m];
    jds.h_iperm = new int[coo.m];
    for (int i = 0; i < coo.m; i++) {
        jds.h_perm[i] = row_pairs[i].second;
        jds.h_iperm[row_pairs[i].second] = i;
    }

    // Build diagonal structure
    std::vector<std::vector<int>> diag_col_idx;
    std::vector<std::vector<FloatType>> diag_values;
    std::vector<int> cur_pos(coo.m, 0);

    for (int diag = 0; diag < row_pairs[0].first; diag++) {
        std::vector<int> col_idx;
        std::vector<FloatType> values;
        
        for (int i = 0; i < coo.m; i++) {
            int orig_row = jds.h_perm[i];
            if (cur_pos[orig_row] < row_lengths[orig_row]) {
                // Find the diag-th non-zero for this row
                int count = 0;
                for (int j = 0; j < coo.nnz; j++) {
                    if (coo.row[j] == orig_row) {
                        if (count == cur_pos[orig_row]) {
                            col_idx.push_back(coo.col[j]);
                            values.push_back(coo.val[j]);
                            cur_pos[orig_row]++;
                            break;
                        }
                        count++;
                    }
                }
            }
        }
        
        if (!col_idx.empty()) {
            diag_col_idx.push_back(col_idx);
            diag_values.push_back(values);
        }
    }

    // Store diagonals in JDS format
    int total_diags = (int)diag_col_idx.size();
    jds.h_col_start = new int[total_diags];
    jds.h_diag_len = new int[total_diags];
    
    int col_pos = 0;
    int total_entries = 0;
    for (int d = 0; d < total_diags; d++) {
        jds.h_col_start[d] = col_pos;
        jds.h_diag_len[d] = (int)diag_col_idx[d].size();
        col_pos += (int)diag_col_idx[d].size();
        total_entries += (int)diag_col_idx[d].size();
    }

    // Flatten diagonal data
    jds.h_col_idx = new int[total_entries];
    jds.h_values = new FloatType[total_entries];
    
    col_pos = 0;
    for (int d = 0; d < total_diags; d++) {
        for (int i = 0; i < (int)diag_col_idx[d].size(); i++) {
            jds.h_col_idx[col_pos] = diag_col_idx[d][i];
            jds.h_values[col_pos] = diag_values[d][i];
            col_pos++;
        }
    }

    // Copy to device
    CUDA_CHECK(cudaMalloc(&jds.d_perm, coo.m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&jds.d_iperm, coo.m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&jds.d_col_start, total_diags * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&jds.d_diag_len, total_diags * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&jds.d_col_idx, total_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&jds.d_values, total_entries * sizeof(FloatType)));

    CUDA_CHECK(cudaMemcpy(jds.d_perm, jds.h_perm, coo.m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(jds.d_iperm, jds.h_iperm, coo.m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(jds.d_col_start, jds.h_col_start, total_diags * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(jds.d_diag_len, jds.h_diag_len, total_diags * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(jds.d_col_idx, jds.h_col_idx, total_entries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(jds.d_values, jds.h_values, total_entries * sizeof(FloatType), cudaMemcpyHostToDevice));

    return jds;
}

// Generate random vector with fixed seed
__global__ void fillRandomVector(FloatType *d_vector, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        d_vector[idx] = curand_uniform(&state);
    }
}

void generateRandomVector(FloatType *d_vector, int size, unsigned int seed) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    fillRandomVector<<<gridSize, blockSize>>>(d_vector, size, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Print matrix statistics
void printMatrixStats(const COOMatrix& coo) {
    double sparsity = 100.0 * (1.0 - (double)coo.nnz / (coo.m * coo.n));
    size_t memory_coo = coo.nnz * (2 * sizeof(int) + sizeof(FloatType));
    
    printf("Matrix Statistics:\n");
    printf("  Dimensions: %d x %d\n", coo.m, coo.n);
    printf("  Non-zeros: %d\n", coo.nnz);
    printf("  Sparsity: %.2f%%\n", sparsity);
    printf("  Avg NNZ per row: %.2f\n", (double)coo.nnz / coo.m);
    printf("  Memory (COO format): %.2f KB\n", memory_coo / 1024.0);
}

// Free CSR matrix
void freeCSRMatrix(CSRMatrix& csr) {
    if (csr.d_row_ptr) CUDA_CHECK(cudaFree(csr.d_row_ptr));
    if (csr.d_col_idx) CUDA_CHECK(cudaFree(csr.d_col_idx));
    if (csr.d_values) CUDA_CHECK(cudaFree(csr.d_values));
    if (csr.h_row_ptr) delete[] csr.h_row_ptr;
    if (csr.h_col_idx) delete[] csr.h_col_idx;
    if (csr.h_values) delete[] csr.h_values;
}

// Free ELL matrix
void freeELLMatrix(ELLMatrix& ell) {
    if (ell.d_col_idx) CUDA_CHECK(cudaFree(ell.d_col_idx));
    if (ell.d_values) CUDA_CHECK(cudaFree(ell.d_values));
    if (ell.h_col_idx) delete[] ell.h_col_idx;
    if (ell.h_values) delete[] ell.h_values;
}

// Free JDS matrix
void freeJDSMatrix(JDSMatrix& jds) {
    if (jds.d_perm) CUDA_CHECK(cudaFree(jds.d_perm));
    if (jds.d_iperm) CUDA_CHECK(cudaFree(jds.d_iperm));
    if (jds.d_col_start) CUDA_CHECK(cudaFree(jds.d_col_start));
    if (jds.d_diag_len) CUDA_CHECK(cudaFree(jds.d_diag_len));
    if (jds.d_col_idx) CUDA_CHECK(cudaFree(jds.d_col_idx));
    if (jds.d_values) CUDA_CHECK(cudaFree(jds.d_values));
    if (jds.h_perm) delete[] jds.h_perm;
    if (jds.h_iperm) delete[] jds.h_iperm;
    if (jds.h_col_start) delete[] jds.h_col_start;
    if (jds.h_diag_len) delete[] jds.h_diag_len;
    if (jds.h_col_idx) delete[] jds.h_col_idx;
    if (jds.h_values) delete[] jds.h_values;
}
