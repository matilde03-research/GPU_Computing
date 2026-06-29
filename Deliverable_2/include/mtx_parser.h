#ifndef MTX_PARSER_H
#define MTX_PARSER_H


#ifdef __cplusplus
extern "C" {
#endif

/* Coordinate (COO) sparse matrix, 0-indexed, single-precision values. */
typedef struct {
    int    rows;     /* number of rows    (M) */
    int    cols;     /* number of columns (N) */
    long   nnz;      /* number of stored nonzeros (after symmetric expansion) */
    int   *row_idx;  /* [nnz] global row indices (0-based) */
    int   *col_idx;  /* [nnz] global column indices (0-based) */
    float *val;      /* [nnz] values */
} COOMatrix;

/*
 * Read a Matrix Market coordinate file into 'A' (host memory).
 */
int mm_read_coo(const char *path, COOMatrix *A);

//Generate a square R-MAT matrix.

int rmat_generate(int scale, int edge_factor, unsigned int seed, COOMatrix *A);

/*
 * Generate a structured finite-difference Laplacian on a regular grid:
 * 7-point in 3D, or 5-point in 2D when nz==1. 
 */
int stencil_generate(int nx, int ny, int nz, COOMatrix *A);

/*
 * Fill x[0..n-1] with reproducible pseudo-random float32 values in [-1, 1).
 * The fixed seed (use 42) guarantees the same vector across runs / rank counts.
 */
void generate_random_vector(float *x, int n, unsigned int seed);

/* Release the buffers held by a COOMatrix. */
void coo_free(COOMatrix *A);

#ifdef __cplusplus
}
#endif

#endif /* MTX_PARSER_H */
