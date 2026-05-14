#ifndef MTX_PARSER_H
#define MTX_PARSER_H

#include <vector>
#include <string>

struct COOMatrix {
    int rows;
    int cols;
    int nnz;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;
};

struct CSRMatrix {
    int rows;
    int cols;
    int nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
};

struct ELLMatrix {
    int rows;
    int cols;
    int nnz;
    int max_nnz_per_row;
    std::vector<int> col_idx;
    std::vector<float> values;
};

struct JDSMatrix {
    int rows;
    int cols;
    int nnz;
    std::vector<int> perm;
    std::vector<int> row_lengths;
    std::vector<int> col_idx;
    std::vector<float> values;
};

COOMatrix parse_mtx_file(const char* filename);
CSRMatrix coo_to_csr(const COOMatrix& coo);
ELLMatrix coo_to_ell(const COOMatrix& coo);
JDSMatrix coo_to_jds(const COOMatrix& coo);
std::vector<float> generate_random_vector(int size, int seed = 42);

#endif
