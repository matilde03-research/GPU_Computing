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
 * Supports field = real | integer | pattern (complex: imaginary part dropped).
 * Supports symmetry = general | symmetric | skew-symmetric (expanded to full).
 * Returns 0 on success, non-zero on failure.  Free with coo_free().
 */
int mm_read_coo(const char *path, COOMatrix *A);

/*
 * Generate a square R-MAT matrix.
 *   scale       -> N = 2^scale rows and columns
 *   edge_factor -> ~edge_factor nonzeros per row (nnz ~= edge_factor * N)
 *   seed        -> RNG seed for reproducibility
 * Duplicate (i,j) edges are kept (they simply accumulate during SpMV, exactly
 * as they do in the CPU reference, so correctness is unaffected).
 * Returns 0 on success.  Free with coo_free().
 */
int rmat_generate(int scale, int edge_factor, unsigned int seed, COOMatrix *A);

/*
 * Generate a structured finite-difference Laplacian on a regular grid:
 * 7-point in 3D, or 5-point in 2D when nz==1. The grid is nx x ny x nz, so
 * N = nx*ny*nz rows. Diagonal = 2*ndim, each existing face neighbor = -1.
 * Unlike R-MAT, this has a *uniform* row degree (good load balance) and
 * *locality* (bounded bandwidth), which makes it well suited to weak-scaling
 * studies. Returns 0 on success; free with coo_free().
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
