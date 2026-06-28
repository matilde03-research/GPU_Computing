#include "mtx_parser.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* ------------------------------------------------------------------------- */
/* Small, platform-independent RNG (xorshift32) so that a fixed seed yields   */
/* the exact same matrix / vector on any machine.                            */
/* ------------------------------------------------------------------------- */
static inline unsigned int xorshift32(unsigned int *state)
{
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* uniform float in [0, 1) */
static inline float next_uniform(unsigned int *state)
{
    return (float)(xorshift32(state) & 0x00FFFFFFu) / (float)0x01000000u;
}

/* ------------------------------------------------------------------------- */
/* Matrix Market reader                                                       */
/* ------------------------------------------------------------------------- */
int mm_read_coo(const char *path, COOMatrix *A)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[mtx_parser] cannot open '%s'\n", path);
        return 1;
    }

    char line[1024];

    /* --- banner line: %%MatrixMarket matrix coordinate <field> <symmetry> --- */
    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "[mtx_parser] empty file\n");
        fclose(f);
        return 1;
    }

    char object[64], format[64], field[64], symmetry[64];
    object[0] = format[0] = field[0] = symmetry[0] = '\0';
    if (sscanf(line, "%%%%MatrixMarket %63s %63s %63s %63s",
               object, format, field, symmetry) < 4) {
        fprintf(stderr, "[mtx_parser] malformed MatrixMarket banner\n");
        fclose(f);
        return 1;
    }

    if (strcmp(format, "coordinate") != 0) {
        fprintf(stderr, "[mtx_parser] only 'coordinate' format is supported "
                        "(got '%s')\n", format);
        fclose(f);
        return 1;
    }

    int is_pattern   = (strcmp(field, "pattern")  == 0);
    int is_complex   = (strcmp(field, "complex")  == 0);
    int is_symmetric = (strcmp(symmetry, "symmetric") == 0 ||
                        strcmp(symmetry, "hermitian") == 0);
    int is_skew      = (strcmp(symmetry, "skew-symmetric") == 0);

    /* --- skip comment lines, then read the size line: M N nnz --- */
    long M = 0, N = 0, nnz_stored = 0;
    do {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "[mtx_parser] unexpected EOF before size line\n");
            fclose(f);
            return 1;
        }
    } while (line[0] == '%');

    if (sscanf(line, "%ld %ld %ld", &M, &N, &nnz_stored) != 3) {
        fprintf(stderr, "[mtx_parser] malformed size line\n");
        fclose(f);
        return 1;
    }

    /* Symmetric/skew matrices store only one triangle -> may double up to 2x. */
    long capacity = (is_symmetric || is_skew) ? 2 * nnz_stored : nnz_stored;
    if (capacity < 1) capacity = 1;

    int   *rows = (int   *)malloc(sizeof(int)   * capacity);
    int   *cols = (int   *)malloc(sizeof(int)   * capacity);
    float *vals = (float *)malloc(sizeof(float) * capacity);
    if (!rows || !cols || !vals) {
        fprintf(stderr, "[mtx_parser] out of memory allocating COO\n");
        free(rows); free(cols); free(vals);
        fclose(f);
        return 1;
    }

    long count = 0;
    for (long k = 0; k < nnz_stored; ++k) {
        long r, c;
        double v = 1.0, vi = 0.0;

        if (is_pattern) {
            if (fscanf(f, "%ld %ld", &r, &c) != 2) {
                fprintf(stderr, "[mtx_parser] short read at entry %ld\n", k);
                break;
            }
        } else if (is_complex) {
            if (fscanf(f, "%ld %ld %lf %lf", &r, &c, &v, &vi) != 4) {
                fprintf(stderr, "[mtx_parser] short read at entry %ld\n", k);
                break;
            }
        } else { /* real or integer */
            if (fscanf(f, "%ld %ld %lf", &r, &c, &v) != 3) {
                fprintf(stderr, "[mtx_parser] short read at entry %ld\n", k);
                break;
            }
        }

        /* Matrix Market is 1-indexed. */
        int ri = (int)(r - 1);
        int ci = (int)(c - 1);
        float fv = (float)v;

        rows[count] = ri;
        cols[count] = ci;
        vals[count] = fv;
        ++count;

        /* Mirror the off-diagonal entry for (skew-)symmetric matrices. */
        if ((is_symmetric || is_skew) && ri != ci) {
            rows[count] = ci;
            cols[count] = ri;
            vals[count] = is_skew ? -fv : fv;
            ++count;
        }
    }

    fclose(f);

    A->rows    = (int)M;
    A->cols    = (int)N;
    A->nnz     = count;
    A->row_idx = rows;
    A->col_idx = cols;
    A->val     = vals;

    return 0;
}

/* ------------------------------------------------------------------------- */
/* R-MAT generator (Graph500-style quadrant probabilities a/b/c/d).           */
/* ------------------------------------------------------------------------- */
int rmat_generate(int scale, int edge_factor, unsigned int seed, COOMatrix *A)
{
    if (scale < 1 || edge_factor < 1) {
        fprintf(stderr, "[mtx_parser] invalid R-MAT parameters\n");
        return 1;
    }

    const long N = 1L << scale;
    const long E = (long)edge_factor * N;

    /* Classic R-MAT partition probabilities (skewed -> power-law structure). */
    const float a = 0.57f, b = 0.19f, c = 0.19f; /* d = 1 - a - b - c = 0.05 */

    int   *rows = (int   *)malloc(sizeof(int)   * E);
    int   *cols = (int   *)malloc(sizeof(int)   * E);
    float *vals = (float *)malloc(sizeof(float) * E);
    if (!rows || !cols || !vals) {
        fprintf(stderr, "[mtx_parser] out of memory allocating R-MAT (E=%ld)\n", E);
        free(rows); free(cols); free(vals);
        return 1;
    }

    unsigned int state = (seed == 0u) ? 0x9E3779B9u : seed; /* xorshift needs !=0 */

    for (long e = 0; e < E; ++e) {
        int r = 0, col = 0;
        for (int s = 0; s < scale; ++s) {
            float p = next_uniform(&state);
            if (p < a) {
                /* top-left quadrant: row bit 0, col bit 0 */
            } else if (p < a + b) {
                col |= (1 << s);              /* top-right  */
            } else if (p < a + b + c) {
                r   |= (1 << s);              /* bottom-left */
            } else {
                r   |= (1 << s);              /* bottom-right */
                col |= (1 << s);
            }
        }
        rows[e] = r;
        cols[e] = col;
        /* nonzero value in [0.5, 1.5) -> keeps magnitudes well away from 0 */
        vals[e] = 0.5f + next_uniform(&state);
    }

    A->rows    = (int)N;
    A->cols    = (int)N;
    A->nnz     = E;
    A->row_idx = rows;
    A->col_idx = cols;
    A->val     = vals;

    return 0;
}

/* ------------------------------------------------------------------------- */
/* Structured finite-difference Laplacian (7-point 3D / 5-point 2D).          */
/* ------------------------------------------------------------------------- */
int stencil_generate(int nx, int ny, int nz, COOMatrix *A)
{
    if (nx < 1 || ny < 1 || nz < 1) {
        fprintf(stderr, "[mtx_parser] invalid stencil dimensions %dx%dx%d\n",
                nx, ny, nz);
        return 1;
    }

    const long N    = (long)nx * ny * nz;
    int        ndim = (nx > 1) + (ny > 1) + (nz > 1);
    if (ndim == 0) ndim = 1;
    const float diag = (float)(2 * ndim);          /* 4 in 2D, 6 in 3D */

    /* at most (2*ndim) off-diagonals + 1 diagonal per row */
    const long cap = N * (long)(2 * ndim + 1);
    int   *rows = (int   *)malloc(sizeof(int)   * cap);
    int   *cols = (int   *)malloc(sizeof(int)   * cap);
    float *vals = (float *)malloc(sizeof(float) * cap);
    if (!rows || !cols || !vals) {
        fprintf(stderr, "[mtx_parser] out of memory allocating stencil (N=%ld)\n", N);
        free(rows); free(cols); free(vals);
        return 1;
    }

    const long plane = (long)nx * ny;              /* z-neighbor stride */
    long cnt = 0;
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                long idx = (long)x + (long)y * nx + (long)z * plane;

                /* diagonal */
                rows[cnt] = (int)idx; cols[cnt] = (int)idx; vals[cnt] = diag; ++cnt;

                /* face neighbors that lie inside the grid */
                if (x > 0)      { rows[cnt]=(int)idx; cols[cnt]=(int)(idx-1);     vals[cnt]=-1.0f; ++cnt; }
                if (x < nx - 1) { rows[cnt]=(int)idx; cols[cnt]=(int)(idx+1);     vals[cnt]=-1.0f; ++cnt; }
                if (y > 0)      { rows[cnt]=(int)idx; cols[cnt]=(int)(idx-nx);    vals[cnt]=-1.0f; ++cnt; }
                if (y < ny - 1) { rows[cnt]=(int)idx; cols[cnt]=(int)(idx+nx);    vals[cnt]=-1.0f; ++cnt; }
                if (z > 0)      { rows[cnt]=(int)idx; cols[cnt]=(int)(idx-plane); vals[cnt]=-1.0f; ++cnt; }
                if (z < nz - 1) { rows[cnt]=(int)idx; cols[cnt]=(int)(idx+plane); vals[cnt]=-1.0f; ++cnt; }
            }
        }
    }

    A->rows    = (int)N;
    A->cols    = (int)N;
    A->nnz     = cnt;
    A->row_idx = rows;
    A->col_idx = cols;
    A->val     = vals;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Reproducible random dense operand vector.                                  */
/* ------------------------------------------------------------------------- */
void generate_random_vector(float *x, int n, unsigned int seed)
{
    unsigned int state = (seed == 0u) ? 0x9E3779B9u : seed;
    for (int i = 0; i < n; ++i) {
        x[i] = 2.0f * next_uniform(&state) - 1.0f; /* [-1, 1) */
    }
}

/* ------------------------------------------------------------------------- */
void coo_free(COOMatrix *A)
{
    if (!A) return;
    free(A->row_idx);
    free(A->col_idx);
    free(A->val);
    A->row_idx = NULL;
    A->col_idx = NULL;
    A->val     = NULL;
    A->nnz     = 0;
}
