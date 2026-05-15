#include "../include/mtx_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <cmath>

COOMatrix readMatrixMarket(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    std::string line;
    int m = 0, n = 0, nnz = 0;
    bool is_symmetric = false;

    while (std::getline(file, line)) {
        if (line[0] == '%') {
            if (line.find("symmetric") != std::string::npos) {
                is_symmetric = true;
            }
            continue;
        }
        std::istringstream iss(line);
        iss >> m >> n >> nnz;
        break;
    }

    if (m == 0 || n == 0 || nnz == 0) {
        fprintf(stderr, "Error: Invalid matrix dimensions\n");
        exit(1);
    }

    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<FloatType> values;

    row_indices.reserve(nnz * (is_symmetric ? 2 : 1));
    col_indices.reserve(nnz * (is_symmetric ? 2 : 1));
    values.reserve(nnz * (is_symmetric ? 2 : 1));

    int actual_nnz = 0;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;

        std::istringstream iss(line);
        int i, j;
        FloatType val = 1.0f;

        if (!(iss >> i >> j)) continue;
        if (!(iss >> val)) {
            val = 1.0f;
        }

        i--; j--;

        if (i < 0 || i >= m || j < 0 || j >= n) {
            fprintf(stderr, "Warning: Index out of bounds (%d, %d)\n", i, j);
            continue;
        }

        row_indices.push_back(i);
        col_indices.push_back(j);
        values.push_back(val);
        actual_nnz++;

        if (is_symmetric && i != j) {
            row_indices.push_back(j);
            col_indices.push_back(i);
            values.push_back(val);
            actual_nnz++;
        }
    }

    file.close();

    return {m, n, actual_nnz, row_indices, col_indices, values};
}

void printMatrixStats(const COOMatrix& coo) {
    printf("Matrix Market file loaded:\n");
    printf("  Dimensions: %d × %d\n", coo.m, coo.n);
    printf("  Non-zeros: %d\n", coo.nnz);
    printf("  Sparsity: %.2f%%\n", 100.0 * (1.0 - (double)coo.nnz / (coo.m * coo.n)));
}

CSRMatrix cooToCSR(const COOMatrix& coo) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> row_ptr(coo.m + 1, 0);
    std::vector<int> col_idx(coo.nnz);
    std::vector<FloatType> values(coo.nnz);

    std::vector<int> indices(coo.nnz);
    for (int i = 0; i < coo.nnz; i++) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&coo](int a, int b) {
        if (coo.row_indices[a] != coo.row_indices[b]) {
            return coo.row_indices[a] < coo.row_indices[b];
        }
        return coo.col_indices[a] < coo.col_indices[b];
    });

    for (int i = 0; i < coo.nnz; i++) {
        int idx = indices[i];
        col_idx[i] = coo.col_indices[idx];
        values[i] = coo.values[idx];
    }

    for (int i = 0; i < coo.nnz; i++) {
        int idx = indices[i];
        int row = coo.row_indices[idx];
        row_ptr[row + 1]++;
    }

    for (int i = 1; i <= coo.m; i++) {
        row_ptr[i] += row_ptr[i - 1];
    }

    // Allocate GPU memory
    int *d_row_ptr, *d_col_idx;
    FloatType *d_values;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (coo.m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, coo.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, coo.nnz * sizeof(FloatType)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (coo.m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), coo.nnz * sizeof(FloatType), cudaMemcpyHostToDevice));

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("CSR conversion time: %.4f ms\n", elapsed);

    CSRMatrix csr;
    csr.m = coo.m;
    csr.n = coo.n;
    csr.nnz = coo.nnz;
    csr.row_ptr = row_ptr;
    csr.col_idx = col_idx;
    csr.values = values;
    csr.d_row_ptr = d_row_ptr;
    csr.d_col_idx = d_col_idx;
    csr.d_values = d_values;

    return csr;
}

ELLMatrix cooToELL(const COOMatrix& coo) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> nnz_per_row(coo.m, 0);
    for (int i = 0; i < coo.nnz; i++) {
        nnz_per_row[coo.row_indices[i]]++;
    }

    int max_row_len = *std::max_element(nnz_per_row.begin(), nnz_per_row.end());

    std::vector<int> col_idx(coo.m * max_row_len, -1);
    std::vector<FloatType> values(coo.m * max_row_len, 0.0f);

    std::vector<int> row_positions(coo.m, 0);

    std::vector<int> indices(coo.nnz);
    for (int i = 0; i < coo.nnz; i++) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&coo](int a, int b) {
        if (coo.row_indices[a] != coo.row_indices[b]) {
            return coo.row_indices[a] < coo.row_indices[b];
        }
        return coo.col_indices[a] < coo.col_indices[b];
    });

    for (int i = 0; i < coo.nnz; i++) {
        int idx = indices[i];
        int row = coo.row_indices[idx];
        int col = coo.col_indices[idx];
        FloatType val = coo.values[idx];

        int pos = row_positions[row];
        col_idx[row * max_row_len + pos] = col;
        values[row * max_row_len + pos] = val;
        row_positions[row]++;
    }

    // Allocate GPU memory
    int *d_col_idx;
    FloatType *d_values;
    CUDA_CHECK(cudaMalloc(&d_col_idx, coo.m * max_row_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, coo.m * max_row_len * sizeof(FloatType)));

    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), coo.m * max_row_len * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), coo.m * max_row_len * sizeof(FloatType), cudaMemcpyHostToDevice));

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("ELL conversion time: %.4f ms\n", elapsed);

    ELLMatrix ell;
    ell.m = coo.m;
    ell.n = coo.n;
    ell.nnz = coo.nnz;
    ell.max_row_len = max_row_len;
    ell.col_idx = col_idx;
    ell.values = values;
    ell.d_col_idx = d_col_idx;
    ell.d_values = d_values;

    return ell;
}

JDSMatrix cooToJDS(const COOMatrix& coo) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> nnz_per_row(coo.m, 0);
    for (int i = 0; i < coo.nnz; i++) {
        nnz_per_row[coo.row_indices[i]]++;
    }

    std::vector<int> perm(coo.m);
    for (int i = 0; i < coo.m; i++) {
        perm[i] = i;
    }

    std::sort(perm.begin(), perm.end(), [&nnz_per_row](int a, int b) {
        return nnz_per_row[a] > nnz_per_row[b];
    });

    std::vector<std::vector<std::pair<int, FloatType>>> row_data(coo.m);
    for (int i = 0; i < coo.nnz; i++) {
        int row = coo.row_indices[i];
        int col = coo.col_indices[i];
        FloatType val = coo.values[i];
        row_data[row].push_back({col, val});
    }

    for (int i = 0; i < coo.m; i++) {
        std::sort(row_data[i].begin(), row_data[i].end());
    }

    std::vector<int> diag_len;
    std::vector<int> col_start;
    std::vector<int> col_idx;
    std::vector<FloatType> values;

    // Find max row length to determine number of diagonals
    int max_diag = 0;
    for (int i = 0; i < coo.m; i++) {
        max_diag = std::max(max_diag, (int)row_data[perm[i]].size());
    }

    col_start.resize(max_diag);
    diag_len.resize(max_diag, 0);

    // Build diagonals
    for (int d = 0; d < max_diag; d++) {
        col_start[d] = col_idx.size();
        for (int i = 0; i < coo.m; i++) {
            if ((int)row_data[perm[i]].size() > d) {
                col_idx.push_back(row_data[perm[i]][d].first);
                values.push_back(row_data[perm[i]][d].second);
                diag_len[d]++;
            }
        }
    }

    // Allocate GPU memory
    int *d_perm, *d_col_start, *d_diag_len, *d_col_idx;
    FloatType *d_values;
    
    CUDA_CHECK(cudaMalloc(&d_perm, coo.m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_start, max_diag * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_diag_len, max_diag * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, coo.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, coo.nnz * sizeof(FloatType)));

    CUDA_CHECK(cudaMemcpy(d_perm, perm.data(), coo.m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_start, col_start.data(), max_diag * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_diag_len, diag_len.data(), max_diag * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), coo.nnz * sizeof(FloatType), cudaMemcpyHostToDevice));

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("JDS conversion time: %.4f ms\n", elapsed);

    JDSMatrix jds;
    jds.m = coo.m;
    jds.n = coo.n;
    jds.nnz = coo.nnz;
    jds.perm = perm;
    jds.col_start = col_start;
    jds.diag_len = diag_len;
    jds.col_idx = col_idx;
    jds.values = values;
    jds.d_perm = d_perm;
    jds.d_col_start = d_col_start;
    jds.d_diag_len = d_diag_len;
    jds.d_col_idx = d_col_idx;
    jds.d_values = d_values;

    return jds;
}

void freeCSRMatrix(CSRMatrix& csr) {
    CUDA_CHECK(cudaFree(csr.d_row_ptr));
    CUDA_CHECK(cudaFree(csr.d_col_idx));
    CUDA_CHECK(cudaFree(csr.d_values));
    csr.row_ptr.clear();
    csr.col_idx.clear();
    csr.values.clear();
}

void freeELLMatrix(ELLMatrix& ell) {
    CUDA_CHECK(cudaFree(ell.d_col_idx));
    CUDA_CHECK(cudaFree(ell.d_values));
    ell.col_idx.clear();
    ell.values.clear();
}

void freeJDSMatrix(JDSMatrix& jds) {
    CUDA_CHECK(cudaFree(jds.d_perm));
    CUDA_CHECK(cudaFree(jds.d_col_start));
    CUDA_CHECK(cudaFree(jds.d_diag_len));
    CUDA_CHECK(cudaFree(jds.d_col_idx));
    CUDA_CHECK(cudaFree(jds.d_values));
    jds.perm.clear();
    jds.col_start.clear();
    jds.diag_len.clear();
    jds.col_idx.clear();
    jds.values.clear();
}

void generateRandomVector(FloatType *d_x, int n, int seed) {
    std::vector<FloatType> h_x(n);
    srand(seed);
    for (int i = 0; i < n; i++) {
        h_x[i] = (FloatType)rand() / RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(FloatType), cudaMemcpyHostToDevice));
}

// CPU baseline: straightforward OpenMP implementation
void csrSpMV_CPU(int m, int n, const int *row_ptr, const int *col_idx, 
                  const FloatType *values, const FloatType *x, FloatType *y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        FloatType sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}

// Validate GPU result against CPU baseline
bool validateResult(int m, const FloatType *gpu_result, const FloatType *cpu_result, double tolerance = 1e-4) {
    bool valid = true;
    int errors = 0;
    const int max_errors_to_print = 5;

    for (int i = 0; i < m; i++) {
        double diff = fabs((double)gpu_result[i] - (double)cpu_result[i]);
        double relative_error = (cpu_result[i] != 0.0) ? diff / fabs((double)cpu_result[i]) : diff;
        
        if (relative_error > tolerance && diff > 1e-8) {
            if (errors < max_errors_to_print) {
                printf("Mismatch at row %d: GPU=%.6e, CPU=%.6e, rel_error=%.6e\n", 
                       i, gpu_result[i], cpu_result[i], relative_error);
            }
            valid = false;
            errors++;
        }
    }

    if (!valid) {
        printf("Validation FAILED: %d mismatches detected (showing first %d)\n", errors, max_errors_to_print);
    } else {
        printf("✓ Validation PASSED: Results match CPU baseline (tolerance: %.2e)\n", tolerance);
    }
    return valid;
}
