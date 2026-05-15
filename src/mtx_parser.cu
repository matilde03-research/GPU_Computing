#include "../include/mtx_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <iostream>

COOMatrix readMatrixMarket(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    int rows = 0, cols = 0, nnz = 0;
    bool is_symmetric = false;

    while (std::getline(file, line)) {
        if (line[0] == '%') {
            if (line.find("symmetric") != std::string::npos) {
                is_symmetric = true;
            }
            continue;
        }
        std::istringstream iss(line);
        iss >> rows >> cols >> nnz;
        break;
    }

    if (rows == 0 || cols == 0 || nnz == 0) {
        std::cerr << "Error: Invalid matrix dimensions" << std::endl;
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

        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            std::cerr << "Warning: Index out of bounds (" << i << ", " << j << ")" << std::endl;
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

    std::cout << "Matrix Market file parsed:" << std::endl;
    std::cout << "  Dimensions: " << rows << " x " << cols << std::endl;
    std::cout << "  Non-zeros: " << actual_nnz << std::endl;
    std::cout << "  Symmetric: " << (is_symmetric ? "yes" : "no") << std::endl;

    return {rows, cols, actual_nnz, row_indices, col_indices, values};
}

CSRMatrix cooToCSR(const COOMatrix& coo) {
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

    return {coo.m, coo.n, coo.nnz, row_ptr, col_idx, values};
}

ELLMatrix cooToELL(const COOMatrix& coo) {
    std::vector<int> nnz_per_row(coo.m, 0);
    for (int i = 0; i < coo.nnz; i++) {
        nnz_per_row[coo.row_indices[i]]++;
    }

    int max_nnz_per_row = *std::max_element(nnz_per_row.begin(), nnz_per_row.end());

    std::vector<int> col_idx(coo.m * max_nnz_per_row, -1);
    std::vector<FloatType> values(coo.m * max_nnz_per_row, 0.0f);

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
        col_idx[row * max_nnz_per_row + pos] = col;
        values[row * max_nnz_per_row + pos] = val;
        row_positions[row]++;
    }

    return {coo.m, coo.n, coo.nnz, max_nnz_per_row, col_idx, values};
}

JDSMatrix cooToJDS(const COOMatrix& coo) {
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

    std::vector<int> col_idx;
    std::vector<FloatType> values;
    std::vector<int> diag_len(coo.m);

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

    std::vector<int> col_start;
    col_start.push_back(0);

    for (int i = 0; i < coo.m; i++) {
        int row = perm[i];
        diag_len[i] = row_data[row].size();
        
        for (int j = 0; j < row_data[row].size(); j++) {
            col_idx.push_back(row_data[row][j].first);
            values.push_back(row_data[row][j].second);
        }
        
        col_start.push_back(col_idx.size());
    }

    return {coo.m, coo.n, coo.nnz, perm, diag_len, col_start, col_idx, values};
}

std::vector<FloatType> generateRandomVector(int size, int seed) {
    std::vector<FloatType> vec(size);
    srand(seed);
    
    for (int i = 0; i < size; i++) {
        vec[i] = (FloatType)rand() / RAND_MAX;
    }
    
    return vec;
}

void printMatrixStats(const COOMatrix& coo) {
    std::cout << "Matrix Statistics:" << std::endl;
    std::cout << "  Dimensions: " << coo.m << " x " << coo.n << std::endl;
    std::cout << "  Non-zeros: " << coo.nnz << std::endl;
    std::cout << "  Density: " << (100.0 * coo.nnz) / (coo.m * coo.n) << "%" << std::endl;
}

void generateRandomVector(FloatType *d_x, int n, int seed) {
    FloatType *h_x = new FloatType[n];
    srand(seed);
    
    for (int i = 0; i < n; i++) {
        h_x[i] = (FloatType)rand() / RAND_MAX;
    }
    
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n * sizeof(FloatType), cudaMemcpyHostToDevice));
    delete[] h_x;
}

void freeCSRMatrix(CSRMatrix& csr) {
    if (csr.d_row_ptr != nullptr) {
        CUDA_CHECK(cudaFree(csr.d_row_ptr));
        csr.d_row_ptr = nullptr;
    }
    if (csr.d_col_idx != nullptr) {
        CUDA_CHECK(cudaFree(csr.d_col_idx));
        csr.d_col_idx = nullptr;
    }
    if (csr.d_values != nullptr) {
        CUDA_CHECK(cudaFree(csr.d_values));
        csr.d_values = nullptr;
    }
}

void freeELLMatrix(ELLMatrix& ell) {
    if (ell.d_col_idx != nullptr) {
        CUDA_CHECK(cudaFree(ell.d_col_idx));
        ell.d_col_idx = nullptr;
    }
    if (ell.d_values != nullptr) {
        CUDA_CHECK(cudaFree(ell.d_values));
        ell.d_values = nullptr;
    }
}

void freeJDSMatrix(JDSMatrix& jds) {
    if (jds.d_perm != nullptr) {
        CUDA_CHECK(cudaFree(jds.d_perm));
        jds.d_perm = nullptr;
    }
    if (jds.d_diag_len != nullptr) {
        CUDA_CHECK(cudaFree(jds.d_diag_len));
        jds.d_diag_len = nullptr;
    }
    if (jds.d_col_start != nullptr) {
        CUDA_CHECK(cudaFree(jds.d_col_start));
        jds.d_col_start = nullptr;
    }
    if (jds.d_col_idx != nullptr) {
        CUDA_CHECK(cudaFree(jds.d_col_idx));
        jds.d_col_idx = nullptr;
    }
    if (jds.d_values != nullptr) {
        CUDA_CHECK(cudaFree(jds.d_values));
        jds.d_values = nullptr;
    }
}

void spmvCPU_CSR(int m, int n, const int *row_ptr, const int *col_idx, 
                 const FloatType *values, const FloatType *x, FloatType *y) {
    for (int i = 0; i < m; i++) {
        y[i] = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            y[i] += values[j] * x[col_idx[j]];
        }
    }
}

void spmvCPU_ELL(int m, int n, int max_row_len, const int *col_idx, 
                 const FloatType *values, const FloatType *x, FloatType *y) {
    for (int i = 0; i < m; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < max_row_len; j++) {
            int col = col_idx[i * max_row_len + j];
            if (col >= 0) {
                y[i] += values[i * max_row_len + j] * x[col];
            }
        }
    }
}

void spmvCPU_JDS(int m, int n, const int *perm, const int *diag_len, const int *col_start,
                 const int *col_idx, const FloatType *values, const FloatType *x, FloatType *y) {
    std::fill(y, y + m, 0.0f);
    
    for (int i = 0; i < m; i++) {
        int row = perm[i];
        int len = diag_len[i];
        
        for (int j = 0; j < len; j++) {
            int idx = col_start[i] + j;
            y[row] += values[idx] * x[col_idx[idx]];
        }
    }
}
