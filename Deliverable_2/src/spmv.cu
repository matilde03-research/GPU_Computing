/*
 * Launch:
 *   mpirun -np <P> ./bin/spmv <matrix.mtx | random> <warmup> <iters> <P>
 *   with P in {1, 2, 4}; one MPI rank is bound to one GPU.
 * =============================================================================
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "mtx_parser.h"

/* --------------------------------------------------------------------------- */
/* Configuration                                                               */
/* --------------------------------------------------------------------------- */
#define VECTOR_SEED      42u    /* fixed RNG seed for the operand vector x      */
#define RMAT_SEED        12345u /* fixed RNG seed for the synthetic matrix      */
#define RMAT_BASE_SCALE  18     /* P=1 -> 2^18 rows; scale grows with P (weak)  */
#define RMAT_EDGE_FACTOR 16     /* avg nonzeros per row for the R-MAT matrix     */

#ifdef NO_CUDA_AWARE_MPI
#define CUDA_AWARE_MPI 0
#else
#define CUDA_AWARE_MPI 1
#endif

/* --------------------------------------------------------------------------- */
/* Error-checking helpers                                                      */
/* --------------------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "[CUDA] %s:%d  %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(_e));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

/* Check a kernel *launch* error. cudaDeviceSynchronize() does NOT report a
 * launch failure (e.g. a binary built for the wrong -arch); cudaGetLastError()
 * right after the launch does. Without this, a failed launch silently produces
 * zeros instead of aborting. */
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

#define CUSPARSE_CHECK(call)                                                    \
    do {                                                                        \
        cusparseStatus_t _s = (call);                                           \
        if (_s != CUSPARSE_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[cuSPARSE] %s:%d  status=%d\n", __FILE__,          \
                    __LINE__, (int)_s);                                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

/* =========================================================================== */
/* CUDA kernels                                                                */
/* =========================================================================== */

/* CSR-vector kernel: one warp cooperates on one (local) row.
 * accumulate==0 -> y[row]  = A*x   (owned-column pass)
 * accumulate!=0 -> y[row] += A*x   (ghost-column pass) */
__global__ void csr_vector_kernel(int n_rows,
                                  const int   *__restrict__ row_ptr,
                                  const int   *__restrict__ col_idx,
                                  const float *__restrict__ vals,
                                  const float *__restrict__ x,
                                  float       *__restrict__ y,
                                  int accumulate)
{
    const int  global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int  warp_id    = global_tid >> 5;   /* /32 */
    const int  lane       = global_tid & 31;

    if (warp_id >= n_rows) return;

    const int row_start = row_ptr[warp_id];
    const int row_end   = row_ptr[warp_id + 1];

    float sum = 0.0f;
    for (int j = row_start + lane; j < row_end; j += 32)
        sum += vals[j] * x[col_idx[j]];

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    if (lane == 0)
        y[warp_id] = accumulate ? (y[warp_id] + sum) : sum;
}

/* Gather owned x entries into a contiguous send buffer. */
__global__ void gather_kernel(float *__restrict__ out,
                              const float *__restrict__ x_local,
                              const int   *__restrict__ idx,
                              int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x_local[idx[i]];
}

/* Scatter received ghost values into the global-indexed x buffer. */
__global__ void scatter_kernel(float *__restrict__ x_global,
                               const float *__restrict__ in,
                               const int   *__restrict__ idx,
                               int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x_global[idx[i]] = in[i];
}

/* Place the rank's owned x entries into the global-indexed x buffer.
 * Owned local entry k maps to global index (rank + k*P). */
__global__ void fill_owned_kernel(float *__restrict__ x_global,
                                  const float *__restrict__ x_local,
                                  int rank, int P, int n_local)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n_local) x_global[(long)rank + (long)k * P] = x_local[k];
}

/* =========================================================================== */
/* Host helpers                                                                */
/* =========================================================================== */

/* Number of indices in [0,total) owned by 'rank' under cyclic partitioning. */
static long count_cyclic(long total, int rank, int P)
{
    if ((long)rank >= total) return 0;
    return (total - 1 - rank) / P + 1;
}

/*
 * Build a CSR representation from COO whose rows are addressed cyclically.
 * The produced row pointer is indexed by LOCAL row (global_row / P); column
 * indices stay GLOBAL.  Passing rank=0, P=1, n_local=n_rows yields a plain
 * full-matrix CSR (used for the rank-0 reference and single-GPU baseline).
 */
static void build_cyclic_csr(const int *grow, const int *gcol, const float *gval,
                             long nnz, int n_local, int /*rank*/, int P,
                             int **row_ptr_out, int **col_idx_out, float **val_out)
{
    int   *row_ptr = (int   *)calloc((size_t)n_local + 1, sizeof(int));
    int   *col_idx = (int   *)malloc(sizeof(int)   * (nnz > 0 ? nnz : 1));
    float *vals    = (float *)malloc(sizeof(float) * (nnz > 0 ? nnz : 1));

    for (long k = 0; k < nnz; ++k) {
        int lr = grow[k] / P;          /* local row index */
        row_ptr[lr + 1]++;
    }
    for (int i = 0; i < n_local; ++i)
        row_ptr[i + 1] += row_ptr[i];

    int *pos = (int *)malloc(sizeof(int) * (size_t)(n_local > 0 ? n_local : 1));
    for (int i = 0; i < n_local; ++i) pos[i] = row_ptr[i];

    for (long k = 0; k < nnz; ++k) {
        int lr = grow[k] / P;
        int p  = pos[lr]++;
        col_idx[p] = gcol[k];
        vals[p]    = gval[k];
    }
    free(pos);

    *row_ptr_out = row_ptr;
    *col_idx_out = col_idx;
    *val_out     = vals;
}

/*
 * Split a local CSR (global column indices) into two CSRs over the SAME rows:
 *   - "owned" : entries whose column is locally owned  (col % P == rank)
 *   - "ghost" : entries whose column is remote         (col % P != rank)
 * Lets the owned-column SpMV run while the ghost halo is still in flight.
 */
static void split_csr_by_owner(const int *row_ptr, const int *col_idx,
                               const float *vals, int n_local, int rank, int P,
                               int **o_rp, int **o_ci, float **o_v,
                               int **g_rp, int **g_ci, float **g_v)
{
    int *orp = (int *)calloc((size_t)n_local + 1, sizeof(int));
    int *grp = (int *)calloc((size_t)n_local + 1, sizeof(int));

    for (int i = 0; i < n_local; ++i)
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            if (col_idx[j] % P == rank) orp[i + 1]++; else grp[i + 1]++;

    for (int i = 0; i < n_local; ++i) { orp[i+1] += orp[i]; grp[i+1] += grp[i]; }

    long o_nnz = (n_local > 0) ? orp[n_local] : 0;
    long g_nnz = (n_local > 0) ? grp[n_local] : 0;
    int   *oci = (int   *)malloc(sizeof(int)   * (o_nnz > 0 ? o_nnz : 1));
    float *ov  = (float *)malloc(sizeof(float) * (o_nnz > 0 ? o_nnz : 1));
    int   *gci = (int   *)malloc(sizeof(int)   * (g_nnz > 0 ? g_nnz : 1));
    float *gv  = (float *)malloc(sizeof(float) * (g_nnz > 0 ? g_nnz : 1));

    int *op = (int *)malloc(sizeof(int) * (size_t)(n_local > 0 ? n_local : 1));
    int *gp = (int *)malloc(sizeof(int) * (size_t)(n_local > 0 ? n_local : 1));
    for (int i = 0; i < n_local; ++i) { op[i] = orp[i]; gp[i] = grp[i]; }

    for (int i = 0; i < n_local; ++i)
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (col_idx[j] % P == rank) { oci[op[i]] = col_idx[j]; ov[op[i]] = vals[j]; op[i]++; }
            else                        { gci[gp[i]] = col_idx[j]; gv[gp[i]] = vals[j]; gp[i]++; }
        }
    free(op); free(gp);

    *o_rp = orp; *o_ci = oci; *o_v = ov;
    *g_rp = grp; *g_ci = gci; *g_v = gv;
}

/* Relative L2 error and max abs error between two host vectors. */
static void compute_error(const float *a, const float *ref, int n,
                          double *l2rel, double *maxabs)
{
    double num = 0.0, den = 0.0, mx = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)ref[i];
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    *l2rel  = (den > 0.0) ? sqrt(num / den) : sqrt(num);
    *maxabs = mx;
}

/* min / mean / max of a scalar across all ranks (result valid on rank 0). */
static void reduce_stats(double v, int P, MPI_Comm comm,
                         double *mn, double *mean, double *mx)
{
    double s;
    MPI_Reduce(&v, mn,   1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&v, mx,   1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&v, &s,   1, MPI_DOUBLE, MPI_SUM, 0, comm);
    *mean = s / P;
}

/* =========================================================================== */
/* Ghost-exchange plan                                                         */
/* =========================================================================== */
typedef struct {
    int  P;
    int  total_send, total_recv;        /* float counts */
    int *send_counts, *send_displs;     /* [P] */
    int *recv_counts, *recv_displs;     /* [P] */

    int   *d_send_local;                /* [total_send] local idx into x_local  */
    int   *d_recv_global;               /* [total_recv] global idx into x_global*/
    float *d_send_buf, *d_recv_buf;     /* device staging                       */

    MPI_Request *reqs;                  /* [2*P] request buffer for Isend/Irecv  */

#if !CUDA_AWARE_MPI
    float *h_send_buf, *h_recv_buf;     /* host staging (non CUDA-aware MPI)     */
#endif
} GhostPlan;

/* One ghost exchange: pack owned values -> point-to-point halo -> scatter.
 * Uses Isend/Irecv to the actual neighbour ranks (much faster than a
 * CUDA-aware Alltoallv on this cluster). Still blocking (Waitall) -- the
 * overlap with computation is added on top of this in the CSR-vector path. */
static void ghost_exchange(GhostPlan *g, float *d_x, float *d_xloc)
{
    if (g->total_send > 0) {
        int b = 256, gr = (g->total_send + b - 1) / b;
        gather_kernel<<<gr, b>>>(g->d_send_buf, d_xloc, g->d_send_local,
                                 g->total_send);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaDeviceSynchronize());   /* pack must finish before MPI reads  */

#if CUDA_AWARE_MPI
    float *sbuf = g->d_send_buf, *rbuf = g->d_recv_buf;   /* device pointers */
#else
    if (g->total_send > 0)
        CUDA_CHECK(cudaMemcpy(g->h_send_buf, g->d_send_buf,
                              sizeof(float) * g->total_send, cudaMemcpyDeviceToHost));
    float *sbuf = g->h_send_buf, *rbuf = g->h_recv_buf;   /* host staging */
#endif

    /* point-to-point halo exchange: talk only to the ranks we share data with */
    int nreq = 0;
    for (int p = 0; p < g->P; ++p)
        if (g->recv_counts[p] > 0)
            MPI_Irecv(rbuf + g->recv_displs[p], g->recv_counts[p], MPI_FLOAT,
                      p, 0, MPI_COMM_WORLD, &g->reqs[nreq++]);
    for (int p = 0; p < g->P; ++p)
        if (g->send_counts[p] > 0)
            MPI_Isend(sbuf + g->send_displs[p], g->send_counts[p], MPI_FLOAT,
                      p, 0, MPI_COMM_WORLD, &g->reqs[nreq++]);
    MPI_Waitall(nreq, g->reqs, MPI_STATUSES_IGNORE);

#if !CUDA_AWARE_MPI
    if (g->total_recv > 0)
        CUDA_CHECK(cudaMemcpy(g->d_recv_buf, g->h_recv_buf,
                              sizeof(float) * g->total_recv, cudaMemcpyHostToDevice));
#endif

    if (g->total_recv > 0) {
        int b = 256, gr = (g->total_recv + b - 1) / b;
        scatter_kernel<<<gr, b>>>(d_x, g->d_recv_buf, g->d_recv_global,
                                  g->total_recv);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* Per-phase timing accumulators (seconds). Used only in profiling mode. */
typedef struct { double pack, owned, wait, scatter, ghost; } PhaseT;

/*
 * CSR-vector SpMV with communication / computation overlap (column split):
 *   pack owned x -> post non-blocking halo (Isend/Irecv)
 *   -> compute OWNED-column SpMV while the halo is in flight   (y  = A_owned x)
 *   -> wait -> scatter ghosts
 *   -> compute GHOST-column SpMV                               (y += A_ghost x)
 * Returns the EXPOSED communication time (the MPI_Waitall duration).
 *
 * If prof != NULL the function inserts a stream sync after each phase and
 * records its duration -- this DISABLES the overlap (phases run serially) and
 * is meant purely for diagnosing where time goes. With prof == NULL the
 * owned-column kernel genuinely overlaps the transfer.
 */
static double spmv_csr_overlap(GhostPlan *g, int n_local,
        const int *d_o_rp, const int *d_o_ci, const float *d_o_v,
        const int *d_g_rp, const int *d_g_ci, const float *d_g_v,
        float *d_x, float *d_xloc, float *d_y,
        cudaStream_t comm_s, cudaStream_t comp_s,
        MPI_Request *reqs, int blk, PhaseT *prof)
{
    double t;

    /* 1. pack owned x values other ranks need */
    t = MPI_Wtime();
    if (g->total_send > 0) {
        int gr = (g->total_send + 255) / 256;
        gather_kernel<<<gr, 256, 0, comm_s>>>(g->d_send_buf, d_xloc,
                                              g->d_send_local, g->total_send);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaStreamSynchronize(comm_s));   /* pack done before MPI reads it */
    if (prof) prof->pack += MPI_Wtime() - t;

#if CUDA_AWARE_MPI
    float *sbuf = g->d_send_buf, *rbuf = g->d_recv_buf;
#else
    if (g->total_send > 0)
        CUDA_CHECK(cudaMemcpy(g->h_send_buf, g->d_send_buf,
                              sizeof(float) * g->total_send, cudaMemcpyDeviceToHost));
    float *sbuf = g->h_send_buf, *rbuf = g->h_recv_buf;
#endif

    /* 2. post non-blocking point-to-point halo */
    int nreq = 0;
    for (int p = 0; p < g->P; ++p)
        if (g->recv_counts[p] > 0)
            MPI_Irecv(rbuf + g->recv_displs[p], g->recv_counts[p], MPI_FLOAT,
                      p, 0, MPI_COMM_WORLD, &reqs[nreq++]);
    for (int p = 0; p < g->P; ++p)
        if (g->send_counts[p] > 0)
            MPI_Isend(sbuf + g->send_displs[p], g->send_counts[p], MPI_FLOAT,
                      p, 0, MPI_COMM_WORLD, &reqs[nreq++]);

    /* 3. owned-column SpMV -- overlaps the transfer (unless profiling) */
    t = MPI_Wtime();
    if (n_local > 0) {
        int gr = (n_local * 32 + blk - 1) / blk;
        csr_vector_kernel<<<gr, blk, 0, comp_s>>>(n_local, d_o_rp, d_o_ci, d_o_v,
                                                  d_x, d_y, 0);
        CUDA_CHECK_KERNEL();
    }
    if (prof) { CUDA_CHECK(cudaStreamSynchronize(comp_s)); prof->owned += MPI_Wtime() - t; }

    /* 4. wait for the halo (exposed communication) */
    t = MPI_Wtime();
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
    double exposed = MPI_Wtime() - t;
    if (prof) prof->wait += exposed;

#if !CUDA_AWARE_MPI
    if (g->total_recv > 0)
        CUDA_CHECK(cudaMemcpy(g->d_recv_buf, g->h_recv_buf,
                              sizeof(float) * g->total_recv, cudaMemcpyHostToDevice));
#endif

    /* 5. scatter ghosts into the global x */
    t = MPI_Wtime();
    if (g->total_recv > 0) {
        int gr = (g->total_recv + 255) / 256;
        scatter_kernel<<<gr, 256, 0, comm_s>>>(d_x, g->d_recv_buf,
                                               g->d_recv_global, g->total_recv);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaStreamSynchronize(comm_s));   /* ghosts present before ghost pass */
    if (prof) prof->scatter += MPI_Wtime() - t;

    /* 6. ghost-column SpMV -- accumulates */
    t = MPI_Wtime();
    if (n_local > 0) {
        int gr = (n_local * 32 + blk - 1) / blk;
        csr_vector_kernel<<<gr, blk, 0, comp_s>>>(n_local, d_g_rp, d_g_ci, d_g_v,
                                                  d_x, d_y, 1);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaStreamSynchronize(comp_s));
    if (prof) prof->ghost += MPI_Wtime() - t;

    return exposed;
}

/* =========================================================================== */
/* Metrics printing (collective: all ranks call it, rank 0 prints)            */
/* =========================================================================== */
static void print_metrics(MPI_Comm comm, int rank, int P, const char *name,
                          long nnz_total,
                          double conv_ms, double total_ms, double comm_ms,
                          double compute_ms,
                          long nnz_local, long commvol_floats, double mem_mb,
                          double T1_ms, double l2rel, double maxabs,
                          int validation_ok)
{
    double conv_mn, conv_mean, conv_mx;
    double tot_mn,  tot_mean,  tot_mx;
    double cm_mn,   cm_mean,   cm_mx;
    double cp_mn,   cp_mean,   cp_mx;
    double nnz_mn,  nnz_mean,  nnz_mx;
    double cv_mn,   cv_mean,   cv_mx;
    double mem_mn,  mem_mean,  mem_mx;

    reduce_stats(conv_ms,             P, comm, &conv_mn, &conv_mean, &conv_mx);
    reduce_stats(total_ms,            P, comm, &tot_mn,  &tot_mean,  &tot_mx);
    reduce_stats(comm_ms,             P, comm, &cm_mn,   &cm_mean,   &cm_mx);
    reduce_stats(compute_ms,          P, comm, &cp_mn,   &cp_mean,   &cp_mx);
    reduce_stats((double)nnz_local,   P, comm, &nnz_mn,  &nnz_mean,  &nnz_mx);
    reduce_stats((double)commvol_floats, P, comm, &cv_mn, &cv_mean, &cv_mx);
    reduce_stats(mem_mb,              P, comm, &mem_mn,  &mem_mean,  &mem_mx);

    if (rank != 0) return;

    /* Wall time per SpMV is dictated by the slowest rank. */
    double spmv_ms  = tot_mx;
    double gflops   = (2.0 * (double)nnz_total) / (spmv_ms * 1.0e-3) / 1.0e9;
    double speedup  = (T1_ms > 0.0) ? (T1_ms / spmv_ms) : 0.0;
    double eff      = (T1_ms > 0.0) ? (speedup / P)     : 0.0;

    printf("\n");
    printf("==================  %s  ==================\n", name);
    printf("  COO->CSR conversion (max over ranks) : %10.4f ms\n", conv_mx);
    printf("  SpMV time per iteration (wall)       : %10.4f ms\n", spmv_ms);
    printf("      - computation (max over ranks)   : %10.4f ms\n", cp_mx);
    printf("      - communication (max over ranks) : %10.4f ms\n", cm_mx);
    printf("  FLOPs per SpMV (2 * nnz)             : %10.3e\n",
           2.0 * (double)nnz_total);
    printf("  Performance                          : %10.3f GFLOP/s\n", gflops);
    if (T1_ms > 0.0) {
        printf("  Single-GPU baseline (T1)             : %10.4f ms\n", T1_ms);
        printf("  Speedup  (T1 / T_P)                  : %10.3f\n", speedup);
        printf("  Efficiency (speedup / P)             : %10.3f\n", eff);
    } else {
        printf("  Single-GPU baseline (T1)             :        n/a (skipped)\n");
    }
    printf("  Validation vs OpenMP CPU             : L2rel=%.3e  maxabs=%.3e  [%s]\n",
           l2rel, maxabs, validation_ok ? "PASS" : "FAIL");
    printf("  -- Load balance  nnz / rank          : min=%.0f  avg=%.1f  max=%.0f\n",
           nnz_mn, nnz_mean, nnz_mx);
    printf("  -- Comm volume   ghosts(float)/rank  : min=%.0f  avg=%.1f  max=%.0f"
           "  (max=%.2f MB)\n",
           cv_mn, cv_mean, cv_mx, cv_mx * sizeof(float) / (1024.0 * 1024.0));
    printf("  -- Memory footprint / rank           : min=%.2f  avg=%.2f  max=%.2f MB\n",
           mem_mn, mem_mean, mem_mx);
    fflush(stdout);
}

/* =========================================================================== */
/* main                                                                        */
/* =========================================================================== */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    /* ----------------------------- arguments ------------------------------ */
    if (argc != 5) {
        if (rank == 0)
            fprintf(stderr,
                "Usage: mpirun -np <P> %s "
                "<matrix.mtx | random_rmat | random_stencil> "
                "<warmup> <iters> <num_gpus(1|2|4)>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const char *input   = argv[1];
    const int   warmup  = atoi(argv[2]);
    const int   iters   = atoi(argv[3]);
    const int   num_gpus = atoi(argv[4]);

    if (num_gpus != P) {
        if (rank == 0)
            fprintf(stderr,
                "[error] num_gpus argument (%d) must equal the number of MPI "
                "ranks (-np %d).\n", num_gpus, P);
        MPI_Finalize();
        return 1;
    }
    if (iters < 1) {
        if (rank == 0) fprintf(stderr, "[error] iters must be >= 1\n");
        MPI_Finalize();
        return 1;
    }

    /* ------------------------- bind one GPU per rank ---------------------- */
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        if (rank == 0) fprintf(stderr, "[error] no CUDA device found\n");
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank % ndev));

    /* ===================================================================== */
    /* 1. Rank 0 acquires the global matrix and builds the operand vector x   */
    /* ===================================================================== */
    COOMatrix A;
    memset(&A, 0, sizeof(A));
    int   Nrows = 0, Ncols = 0;
    long  nnz_total = 0;
    float *x_full = NULL;            /* full operand vector (rank 0 only)      */
    /* synthetic-matrix selectors ("random" kept as an alias for R-MAT) */
    int   is_rmat    = (strcmp(input, "random_rmat") == 0) ||
                       (strcmp(input, "random") == 0);
    int   is_stencil = (strcmp(input, "random_stencil") == 0);

    if (rank == 0) {
        if (is_rmat) {
            /* weak scaling: grow the matrix so the per-rank size stays fixed */
            int add = 0; while ((1 << add) < P) ++add;   /* ceil(log2 P) */
            int scale = RMAT_BASE_SCALE + add;
            printf("# Generating R-MAT matrix: scale=%d (N=%d), edge_factor=%d\n",
                   scale, 1 << scale, RMAT_EDGE_FACTOR);
            if (rmat_generate(scale, RMAT_EDGE_FACTOR, RMAT_SEED, &A) != 0) {
                fprintf(stderr, "[error] R-MAT generation failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else if (is_stencil) {
            /* weak scaling: grow the grid so the per-rank size stays ~fixed.
             * Near-cubic 3D grid with the same target N as the R-MAT case. */
            long target = (long)(1 << RMAT_BASE_SCALE) * P;
            int  n = (int)lround(cbrt((double)target));
            if (n < 2) n = 2;
            printf("# Generating 3D 7-point stencil: %d x %d x %d (N=%ld)\n",
                   n, n, n, (long)n * n * n);
            if (stencil_generate(n, n, n, &A) != 0) {
                fprintf(stderr, "[error] stencil generation failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            printf("# Reading Matrix Market file: %s\n", input);
            if (mm_read_coo(input, &A) != 0) {
                fprintf(stderr, "[error] could not read matrix '%s'\n", input);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        Nrows = A.rows;
        Ncols = A.cols;
        nnz_total = A.nnz;

        x_full = (float *)malloc(sizeof(float) * (Ncols > 0 ? Ncols : 1));
        generate_random_vector(x_full, Ncols, VECTOR_SEED);

        printf("# Matrix: %d x %d, nnz = %ld  | P = %d, warmup = %d, iters = %d\n",
               Nrows, Ncols, nnz_total, P, warmup, iters);
        printf("# CUDA-aware MPI: %s\n", CUDA_AWARE_MPI ? "yes" : "no (host-staged)");
        fflush(stdout);
    }

    MPI_Bcast(&Nrows,     1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&Ncols,     1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz_total, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    const int n_local   = (int)count_cyclic(Nrows, rank, P);   /* local rows   */
    const int n_local_x = (int)count_cyclic(Ncols, rank, P);   /* owned x size */

    /* ===================================================================== */
    /* 2. Distribute the matrix (cyclic by row) and x (cyclic by index)       */
    /* ===================================================================== */
    int *send_nnz = NULL, *send_disp = NULL;
    int *pack_row = NULL, *pack_col = NULL;  float *pack_val = NULL;
    int *xs_cnt   = NULL, *xs_disp  = NULL;  float *pack_x   = NULL;

    if (rank == 0) {
        send_nnz  = (int *)calloc(P, sizeof(int));
        send_disp = (int *)malloc(sizeof(int) * P);
        for (long k = 0; k < A.nnz; ++k) send_nnz[A.row_idx[k] % P]++;
        send_disp[0] = 0;
        for (int p = 1; p < P; ++p) send_disp[p] = send_disp[p-1] + send_nnz[p-1];

        pack_row = (int   *)malloc(sizeof(int)   * A.nnz);
        pack_col = (int   *)malloc(sizeof(int)   * A.nnz);
        pack_val = (float *)malloc(sizeof(float) * A.nnz);
        int *off = (int *)malloc(sizeof(int) * P);
        memcpy(off, send_disp, sizeof(int) * P);
        for (long k = 0; k < A.nnz; ++k) {
            int d = A.row_idx[k] % P;
            int p = off[d]++;
            pack_row[p] = A.row_idx[k];
            pack_col[p] = A.col_idx[k];
            pack_val[p] = A.val[k];
        }
        free(off);

        /* operand vector buckets */
        xs_cnt  = (int *)calloc(P, sizeof(int));
        xs_disp = (int *)malloc(sizeof(int) * P);
        for (int j = 0; j < Ncols; ++j) xs_cnt[j % P]++;
        xs_disp[0] = 0;
        for (int p = 1; p < P; ++p) xs_disp[p] = xs_disp[p-1] + xs_cnt[p-1];
        pack_x = (float *)malloc(sizeof(float) * (Ncols > 0 ? Ncols : 1));
        int *xoff = (int *)malloc(sizeof(int) * P);
        memcpy(xoff, xs_disp, sizeof(int) * P);
        for (int j = 0; j < Ncols; ++j) pack_x[xoff[j % P]++] = x_full[j];
        free(xoff);
    }

    /* scatter matrix nonzeros */
    int my_nnz = 0;
    MPI_Scatter(send_nnz, 1, MPI_INT, &my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int   *loc_row = (int   *)malloc(sizeof(int)   * (my_nnz > 0 ? my_nnz : 1));
    int   *loc_col = (int   *)malloc(sizeof(int)   * (my_nnz > 0 ? my_nnz : 1));
    float *loc_val = (float *)malloc(sizeof(float) * (my_nnz > 0 ? my_nnz : 1));
    MPI_Scatterv(pack_row, send_nnz, send_disp, MPI_INT,
                 loc_row, my_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(pack_col, send_nnz, send_disp, MPI_INT,
                 loc_col, my_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(pack_val, send_nnz, send_disp, MPI_FLOAT,
                 loc_val, my_nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* scatter operand vector slice */
    float *x_owned = (float *)malloc(sizeof(float) * (n_local_x > 0 ? n_local_x : 1));
    MPI_Scatterv(pack_x, xs_cnt, xs_disp, MPI_FLOAT,
                 x_owned, n_local_x, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* ===================================================================== */
    /* 3. Local COO -> CSR  (timed)                                           */
    /* ===================================================================== */
    int *h_row_ptr = NULL, *h_col_idx = NULL;  float *h_val = NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double conv_t0 = MPI_Wtime();
    build_cyclic_csr(loc_row, loc_col, loc_val, my_nnz, n_local, rank, P,
                     &h_row_ptr, &h_col_idx, &h_val);
    double conv_ms = (MPI_Wtime() - conv_t0) * 1000.0;

    /* ===================================================================== */
    /* 4. Build the ghost-exchange plan (which remote x entries are needed)   */
    /* ===================================================================== */
    GhostPlan g;
    memset(&g, 0, sizeof(g));
    g.P = P;
    g.recv_counts = (int *)calloc(P, sizeof(int));
    g.recv_displs = (int *)malloc(sizeof(int) * P);
    g.send_counts = (int *)calloc(P, sizeof(int));
    g.send_displs = (int *)malloc(sizeof(int) * P);
    g.reqs        = (MPI_Request *)malloc(sizeof(MPI_Request) * (2 * P));

    /* mark which global columns are referenced and remote */
    char *needed = (char *)calloc((size_t)(Ncols > 0 ? Ncols : 1), 1);
    for (int k = 0; k < my_nnz; ++k) {
        int c = h_col_idx[k];        /* == loc_col, but read from CSR for clarity */
        if (c % P != rank) needed[c] = 1;
    }
    for (int c = 0; c < Ncols; ++c)
        if (needed[c]) g.recv_counts[c % P]++;
    g.recv_displs[0] = 0;
    for (int p = 1; p < P; ++p)
        g.recv_displs[p] = g.recv_displs[p-1] + g.recv_counts[p-1];
    g.total_recv = g.recv_displs[P-1] + g.recv_counts[P-1];

    int *recv_global = (int *)malloc(sizeof(int) * (g.total_recv > 0 ? g.total_recv : 1));
    {
        int *off = (int *)malloc(sizeof(int) * P);
        memcpy(off, g.recv_displs, sizeof(int) * P);
        for (int c = 0; c < Ncols; ++c)
            if (needed[c]) { int o = c % P; recv_global[off[o]++] = c; }
        free(off);
    }
    free(needed);

    /* exchange counts, then the index lists, to learn what we must send */
    MPI_Alltoall(g.recv_counts, 1, MPI_INT, g.send_counts, 1, MPI_INT,
                 MPI_COMM_WORLD);
    g.send_displs[0] = 0;
    for (int p = 1; p < P; ++p)
        g.send_displs[p] = g.send_displs[p-1] + g.send_counts[p-1];
    g.total_send = g.send_displs[P-1] + g.send_counts[P-1];

    int *send_global = (int *)malloc(sizeof(int) * (g.total_send > 0 ? g.total_send : 1));
    MPI_Alltoallv(recv_global, g.recv_counts, g.recv_displs, MPI_INT,
                  send_global, g.send_counts, g.send_displs, MPI_INT,
                  MPI_COMM_WORLD);

    /* global -> owned-local index for packing from x_owned */
    int *send_local = (int *)malloc(sizeof(int) * (g.total_send > 0 ? g.total_send : 1));
    for (int m = 0; m < g.total_send; ++m) send_local[m] = send_global[m] / P;
    free(send_global);

    /* ===================================================================== */
    /* 5. Upload everything to the GPU                                        */
    /* ===================================================================== */
    int   *d_row_ptr = NULL, *d_col_idx = NULL;  float *d_val = NULL;
    float *d_x = NULL, *d_xloc = NULL, *d_y = NULL;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, sizeof(int) * ((size_t)n_local + 1)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, sizeof(int) * (size_t)(my_nnz > 0 ? my_nnz : 1)));
    CUDA_CHECK(cudaMalloc(&d_val,     sizeof(float) * (size_t)(my_nnz > 0 ? my_nnz : 1)));
    CUDA_CHECK(cudaMalloc(&d_x,       sizeof(float) * (size_t)(Ncols > 0 ? Ncols : 1)));
    CUDA_CHECK(cudaMalloc(&d_xloc,    sizeof(float) * (size_t)(n_local_x > 0 ? n_local_x : 1)));
    CUDA_CHECK(cudaMalloc(&d_y,       sizeof(float) * (size_t)(n_local > 0 ? n_local : 1)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, sizeof(int) * ((size_t)n_local + 1),
                          cudaMemcpyHostToDevice));
    if (my_nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx, sizeof(int) * my_nnz,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_val, h_val, sizeof(float) * my_nnz,
                              cudaMemcpyHostToDevice));
    }
    if (n_local_x > 0)
        CUDA_CHECK(cudaMemcpy(d_xloc, x_owned, sizeof(float) * n_local_x,
                              cudaMemcpyHostToDevice));

    /* --- owned/ghost column split for the CSR-vector comm/compute overlap --- */
    int   *ho_rp, *ho_ci, *hg_rp, *hg_ci;  float *ho_v, *hg_v;
    split_csr_by_owner(h_row_ptr, h_col_idx, h_val, n_local, rank, P,
                       &ho_rp, &ho_ci, &ho_v, &hg_rp, &hg_ci, &hg_v);
    long o_nnz = (n_local > 0) ? ho_rp[n_local] : 0;
    long g_nnz = (n_local > 0) ? hg_rp[n_local] : 0;

    int   *d_o_rp=NULL,*d_o_ci=NULL,*d_g_rp=NULL,*d_g_ci=NULL;  float *d_o_v=NULL,*d_g_v=NULL;
    CUDA_CHECK(cudaMalloc(&d_o_rp, sizeof(int)   * ((size_t)n_local + 1)));
    CUDA_CHECK(cudaMalloc(&d_g_rp, sizeof(int)   * ((size_t)n_local + 1)));
    CUDA_CHECK(cudaMalloc(&d_o_ci, sizeof(int)   * (size_t)(o_nnz > 0 ? o_nnz : 1)));
    CUDA_CHECK(cudaMalloc(&d_o_v,  sizeof(float) * (size_t)(o_nnz > 0 ? o_nnz : 1)));
    CUDA_CHECK(cudaMalloc(&d_g_ci, sizeof(int)   * (size_t)(g_nnz > 0 ? g_nnz : 1)));
    CUDA_CHECK(cudaMalloc(&d_g_v,  sizeof(float) * (size_t)(g_nnz > 0 ? g_nnz : 1)));
    CUDA_CHECK(cudaMemcpy(d_o_rp, ho_rp, sizeof(int) * ((size_t)n_local + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g_rp, hg_rp, sizeof(int) * ((size_t)n_local + 1), cudaMemcpyHostToDevice));
    if (o_nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_o_ci, ho_ci, sizeof(int)   * o_nnz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_o_v,  ho_v,  sizeof(float) * o_nnz, cudaMemcpyHostToDevice));
    }
    if (g_nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_g_ci, hg_ci, sizeof(int)   * g_nnz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g_v,  hg_v,  sizeof(float) * g_nnz, cudaMemcpyHostToDevice));
    }
    free(ho_rp); free(ho_ci); free(ho_v); free(hg_rp); free(hg_ci); free(hg_v);

    /* x_global starts at 0; owned entries placed once (x is constant here). */
    CUDA_CHECK(cudaMemset(d_x, 0, sizeof(float) * (size_t)(Ncols > 0 ? Ncols : 1)));
    if (n_local_x > 0) {
        int b = 256, gr = (n_local_x + b - 1) / b;
        fill_owned_kernel<<<gr, b>>>(d_x, d_xloc, rank, P, n_local_x);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* --- GPU-side diagnostics for tiny matrices (read data back off the GPU) - */
    if (rank == 0 && Ncols <= 64 && my_nnz <= 64) {
        float dxg[64], dvg[64], dlg[64];
        CUDA_CHECK(cudaMemcpy(dxg, d_x,    sizeof(float) * Ncols,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dvg, d_val,  sizeof(float) * (my_nnz > 0 ? my_nnz : 1),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dlg, d_xloc, sizeof(float) * (n_local_x > 0 ? n_local_x : 1),
                              cudaMemcpyDeviceToHost));
        printf("  [gpu] n_local=%d n_local_x=%d my_nnz=%d\n", n_local, n_local_x, my_nnz);
        printf("  [gpu] d_xloc readback:");
        for (int i = 0; i < n_local_x; ++i) printf(" %.4f", dlg[i]);
        printf("\n  [gpu] d_x    readback:");
        for (int i = 0; i < Ncols; ++i) printf(" %.4f", dxg[i]);
        printf("\n  [gpu] d_val  readback:");
        for (int i = 0; i < my_nnz; ++i) printf(" %.4f", dvg[i]);
        printf("\n");
        fflush(stdout);
    }

    /* ghost-plan device buffers */
    CUDA_CHECK(cudaMalloc(&g.d_send_local,  sizeof(int)   * (g.total_send > 0 ? g.total_send : 1)));
    CUDA_CHECK(cudaMalloc(&g.d_recv_global, sizeof(int)   * (g.total_recv > 0 ? g.total_recv : 1)));
    CUDA_CHECK(cudaMalloc(&g.d_send_buf,    sizeof(float) * (g.total_send > 0 ? g.total_send : 1)));
    CUDA_CHECK(cudaMalloc(&g.d_recv_buf,    sizeof(float) * (g.total_recv > 0 ? g.total_recv : 1)));
    if (g.total_send > 0)
        CUDA_CHECK(cudaMemcpy(g.d_send_local, send_local, sizeof(int) * g.total_send,
                              cudaMemcpyHostToDevice));
    if (g.total_recv > 0)
        CUDA_CHECK(cudaMemcpy(g.d_recv_global, recv_global, sizeof(int) * g.total_recv,
                              cudaMemcpyHostToDevice));
#if !CUDA_AWARE_MPI
    g.h_send_buf = (float *)malloc(sizeof(float) * (g.total_send > 0 ? g.total_send : 1));
    g.h_recv_buf = (float *)malloc(sizeof(float) * (g.total_recv > 0 ? g.total_recv : 1));
#endif

    /* device memory footprint for this rank (CSR + vectors + halo buffers) */
    double mem_mb = (double)(
        sizeof(int)   * ((size_t)n_local + 1) +
        sizeof(int)   * (size_t)my_nnz +
        sizeof(float) * (size_t)my_nnz +
        sizeof(float) * (size_t)Ncols +
        sizeof(float) * (size_t)n_local_x +
        sizeof(float) * (size_t)n_local +
        sizeof(int)   * (size_t)g.total_send +
        sizeof(int)   * (size_t)g.total_recv +
        sizeof(float) * (size_t)g.total_send +
        sizeof(float) * (size_t)g.total_recv +
        /* owned/ghost split CSRs for the overlap (extra row_ptrs + nnz copy) */
        sizeof(int)   * 2 * ((size_t)n_local + 1) +
        sizeof(int)   * (size_t)my_nnz +
        sizeof(float) * (size_t)my_nnz
    ) / (1024.0 * 1024.0);

    /* ===================================================================== */
    /* OpenMP CPU reference (rank 0) + on-device full-matrix baseline setup    */
    /* ===================================================================== */
    float *y_ref = NULL;                 /* reference result (rank 0)          */
    int   *fr_rp = NULL, *fr_ci = NULL;  float *fr_v = NULL; /* full CSR (rank0)*/
    int   *d_frp = NULL, *d_fci = NULL;  float *d_fv = NULL;
    float *d_fx = NULL, *d_fy = NULL;
    int    baseline_ok = 0;              /* did the single-GPU baseline fit?   */

    if (rank == 0) {
        build_cyclic_csr(A.row_idx, A.col_idx, A.val, A.nnz, Nrows, 0, 1,
                         &fr_rp, &fr_ci, &fr_v);
        y_ref = (float *)malloc(sizeof(float) * (Nrows > 0 ? Nrows : 1));

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nrows; ++i) {
            float s = 0.0f;
            for (int p = fr_rp[i]; p < fr_rp[i + 1]; ++p)
                s += fr_v[p] * x_full[fr_ci[p]];
            y_ref[i] = s;
        }

        /* try to place the full matrix on rank 0's GPU for a T1 baseline */
        cudaError_t e1 = cudaMalloc(&d_frp, sizeof(int) * ((size_t)Nrows + 1));
        cudaError_t e2 = cudaMalloc(&d_fci, sizeof(int) * (size_t)(A.nnz > 0 ? A.nnz : 1));
        cudaError_t e3 = cudaMalloc(&d_fv,  sizeof(float) * (size_t)(A.nnz > 0 ? A.nnz : 1));
        cudaError_t e4 = cudaMalloc(&d_fx,  sizeof(float) * (size_t)(Ncols > 0 ? Ncols : 1));
        cudaError_t e5 = cudaMalloc(&d_fy,  sizeof(float) * (size_t)(Nrows > 0 ? Nrows : 1));
        if (e1==cudaSuccess && e2==cudaSuccess && e3==cudaSuccess &&
            e4==cudaSuccess && e5==cudaSuccess) {
            CUDA_CHECK(cudaMemcpy(d_frp, fr_rp, sizeof(int) * ((size_t)Nrows + 1),
                                  cudaMemcpyHostToDevice));
            if (A.nnz > 0) {
                CUDA_CHECK(cudaMemcpy(d_fci, fr_ci, sizeof(int) * A.nnz, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_fv,  fr_v,  sizeof(float) * A.nnz, cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaMemcpy(d_fx, x_full, sizeof(float) * (Ncols > 0 ? Ncols : 1),
                                  cudaMemcpyHostToDevice));
            baseline_ok = 1;
        } else {
            /* not enough memory (can happen for large weak-scaling cases) */
            cudaGetLastError();
            if (d_frp) cudaFree(d_frp);
            if (d_fci) cudaFree(d_fci);
            if (d_fv)  cudaFree(d_fv);
            if (d_fx)  cudaFree(d_fx);
            if (d_fy)  cudaFree(d_fy);
            d_frp = d_fci = NULL; d_fv = d_fx = d_fy = NULL;
        }
    }

    /* buffers to gather the distributed result on rank 0 */
    int   *ycounts = NULL, *ydispls = NULL;
    float *y_all = NULL, *y_dist = NULL;
    if (rank == 0) {
        ycounts = (int *)malloc(sizeof(int) * P);
        ydispls = (int *)malloc(sizeof(int) * P);
        for (int p = 0; p < P; ++p) ycounts[p] = (int)count_cyclic(Nrows, p, P);
        ydispls[0] = 0;
        for (int p = 1; p < P; ++p) ydispls[p] = ydispls[p-1] + ycounts[p-1];
        y_all  = (float *)malloc(sizeof(float) * (Nrows > 0 ? Nrows : 1));
        y_dist = (float *)malloc(sizeof(float) * (Nrows > 0 ? Nrows : 1));
    }
    float *h_yloc = (float *)malloc(sizeof(float) * (n_local > 0 ? n_local : 1));

    const int blk = 128;                             /* 4 warps per block      */

    /* ===================================================================== */
    /* KERNEL 1: CSR-vector (one warp per row)                                */
    /* ===================================================================== */
    {
        /* streams for the overlap; reuse g.reqs for the non-blocking halo */
        cudaStream_t comm_s, comp_s;
        CUDA_CHECK(cudaStreamCreate(&comm_s));
        CUDA_CHECK(cudaStreamCreate(&comp_s));

        /* warm-up (overlapped) */
        for (int w = 0; w < warmup; ++w)
            spmv_csr_overlap(&g, n_local, d_o_rp, d_o_ci, d_o_v,
                             d_g_rp, d_g_ci, d_g_v, d_x, d_xloc, d_y,
                             comm_s, comp_s, g.reqs, blk, NULL);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* timed loop: owned-column SpMV overlaps the halo exchange */
        double exposed_acc = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it)
            exposed_acc += spmv_csr_overlap(&g, n_local, d_o_rp, d_o_ci, d_o_v,
                                            d_g_rp, d_g_ci, d_g_v, d_x, d_xloc, d_y,
                                            comm_s, comp_s, g.reqs, blk, NULL);
        MPI_Barrier(MPI_COMM_WORLD);
        double total_ms   = (MPI_Wtime() - t0) * 1000.0 / iters;
        double comm_ms    = exposed_acc * 1000.0 / iters;   /* exposed (un-hidden) comm */
        double compute_ms = 0.0;  /* pure CSR kernel time; set from the profile below */

        /* gather + validate */
        if (n_local > 0)
            CUDA_CHECK(cudaMemcpy(h_yloc, d_y, sizeof(float) * n_local,
                                  cudaMemcpyDeviceToHost));
        MPI_Gatherv(h_yloc, n_local, MPI_FLOAT, y_all, ycounts, ydispls,
                    MPI_FLOAT, 0, MPI_COMM_WORLD);

        double l2rel = 0.0, maxabs = 0.0; int ok = 1;
        if (rank == 0) {
            for (int p = 0; p < P; ++p)
                for (int k = 0; k < ycounts[p]; ++k)
                    y_dist[p + k * P] = y_all[ydispls[p] + k];
            compute_error(y_dist, y_ref, Nrows, &l2rel, &maxabs);
            ok = (l2rel < 1e-3);
            if (Nrows <= 32) {                    /* small-matrix diagnostics */
                int m  = (Nrows < 8) ? Nrows : 8;
                int xn = (Ncols < 8) ? Ncols : 8;
                printf("  [debug] x[0..%d):", xn);
                for (int i = 0; i < xn; ++i) printf(" %.4f", x_full[i]);
                printf("\n  [debug]  i :       y_dist          y_ref\n");
                for (int i = 0; i < m; ++i)
                    printf("  [debug] %2d : %14.6f %14.6f\n", i, y_dist[i], y_ref[i]);
            }
        }

        /* per-phase profile: separate loop with per-phase syncs (DISABLES the
         * overlap) purely to show where time goes. Reduced as max over ranks. */
        {
            PhaseT prof = {0, 0, 0, 0, 0};
            int pit = (iters < 20) ? iters : 20;
            for (int it = 0; it < pit; ++it)
                spmv_csr_overlap(&g, n_local, d_o_rp, d_o_ci, d_o_v,
                                 d_g_rp, d_g_ci, d_g_v, d_x, d_xloc, d_y,
                                 comm_s, comp_s, g.reqs, blk, &prof);
            double in[5] = { prof.pack/pit, prof.owned/pit, prof.wait/pit,
                             prof.scatter/pit, prof.ghost/pit };
            double mx[5];
            MPI_Reduce(in, mx, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0)
                printf("  [profile max/iter ms] pack=%.4f owned=%.4f wait=%.4f "
                       "scatter=%.4f ghost=%.4f  (serial sum=%.4f)\n",
                       mx[0]*1e3, mx[1]*1e3, mx[2]*1e3, mx[3]*1e3, mx[4]*1e3,
                       (mx[0]+mx[1]+mx[2]+mx[3]+mx[4])*1e3);

            /* report the actual CSR kernel time (owned + ghost) as "computation" */
            compute_ms = (prof.owned + prof.ghost) / pit * 1000.0;
        }

        /* single-GPU baseline (rank 0) */
        double T1_ms = 0.0;
        if (rank == 0 && baseline_ok) {
            int fgrid = (Nrows * 32 + blk - 1) / blk;
            for (int w = 0; w < warmup; ++w)
                csr_vector_kernel<<<fgrid, blk>>>(Nrows, d_frp, d_fci, d_fv, d_fx, d_fy, 0);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
            double b0 = MPI_Wtime();
            for (int it = 0; it < iters; ++it)
                csr_vector_kernel<<<fgrid, blk>>>(Nrows, d_frp, d_fci, d_fv, d_fx, d_fy, 0);
            CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaDeviceSynchronize());
            T1_ms = (MPI_Wtime() - b0) * 1000.0 / iters;
        }

        print_metrics(MPI_COMM_WORLD, rank, P, "CSR-VECTOR KERNEL", nnz_total,
                      conv_ms, total_ms, comm_ms, compute_ms,
                      my_nnz, g.total_recv, mem_mb, T1_ms, l2rel, maxabs, ok);

        cudaStreamDestroy(comm_s);
        cudaStreamDestroy(comp_s);
    }

    /* ===================================================================== */
    /* KERNEL 2: cuSPARSE generic SpMV                                        */
    /* ===================================================================== */
    {
        cusparseHandle_t handle;
        CUSPARSE_CHECK(cusparseCreate(&handle));

        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecX, vecY;
        const float alpha = 1.0f, beta = 0.0f;

        CUSPARSE_CHECK(cusparseCreateCsr(&matA, n_local, Ncols, my_nnz,
                                         d_row_ptr, d_col_idx, d_val,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, Ncols, d_x, CUDA_R_32F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, n_local, d_y, CUDA_R_32F));

        size_t buf_sz = 0; void *d_buf = NULL;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
                       CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
                       &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_sz));
        if (buf_sz > 0) CUDA_CHECK(cudaMalloc(&d_buf, buf_sz));

        double mem_mb_cs = mem_mb + (double)buf_sz / (1024.0 * 1024.0);

        /* warm-up */
        for (int w = 0; w < warmup; ++w) {
            ghost_exchange(&g, d_x, d_xloc);
            if (n_local > 0)
                CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                               CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        double comm_acc = 0.0, comp_acc = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int it = 0; it < iters; ++it) {
            double c0 = MPI_Wtime();
            ghost_exchange(&g, d_x, d_xloc);
            double c1 = MPI_Wtime();
            comm_acc += (c1 - c0);

            double k0 = MPI_Wtime();
            if (n_local > 0)
                CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                               CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
            CUDA_CHECK(cudaDeviceSynchronize());
            double k1 = MPI_Wtime();
            comp_acc += (k1 - k0);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double total_ms   = (MPI_Wtime() - t0) * 1000.0 / iters;
        double comm_ms    = comm_acc * 1000.0 / iters;
        double compute_ms = comp_acc * 1000.0 / iters;

        if (n_local > 0)
            CUDA_CHECK(cudaMemcpy(h_yloc, d_y, sizeof(float) * n_local,
                                  cudaMemcpyDeviceToHost));
        MPI_Gatherv(h_yloc, n_local, MPI_FLOAT, y_all, ycounts, ydispls,
                    MPI_FLOAT, 0, MPI_COMM_WORLD);

        double l2rel = 0.0, maxabs = 0.0; int ok = 1;
        if (rank == 0) {
            for (int p = 0; p < P; ++p)
                for (int k = 0; k < ycounts[p]; ++k)
                    y_dist[p + k * P] = y_all[ydispls[p] + k];
            compute_error(y_dist, y_ref, Nrows, &l2rel, &maxabs);
            ok = (l2rel < 1e-3);
            if (Nrows <= 32) {                    /* small-matrix diagnostics */
                int m  = (Nrows < 8) ? Nrows : 8;
                int xn = (Ncols < 8) ? Ncols : 8;
                printf("  [debug] x[0..%d):", xn);
                for (int i = 0; i < xn; ++i) printf(" %.4f", x_full[i]);
                printf("\n  [debug]  i :       y_dist          y_ref\n");
                for (int i = 0; i < m; ++i)
                    printf("  [debug] %2d : %14.6f %14.6f\n", i, y_dist[i], y_ref[i]);
            }
        }

        /* single-GPU baseline (rank 0) with cuSPARSE on the full matrix */
        double T1_ms = 0.0;
        if (rank == 0 && baseline_ok) {
            cusparseSpMatDescr_t matF;
            cusparseDnVecDescr_t vXF, vYF;
            CUSPARSE_CHECK(cusparseCreateCsr(&matF, Nrows, Ncols, A.nnz,
                                             d_frp, d_fci, d_fv,
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
            CUSPARSE_CHECK(cusparseCreateDnVec(&vXF, Ncols, d_fx, CUDA_R_32F));
            CUSPARSE_CHECK(cusparseCreateDnVec(&vYF, Nrows, d_fy, CUDA_R_32F));
            size_t bsz = 0; void *fbuf = NULL;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
                           CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matF, vXF,
                           &beta, vYF, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bsz));
            if (bsz > 0) CUDA_CHECK(cudaMalloc(&fbuf, bsz));
            for (int w = 0; w < warmup; ++w)
                cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matF,
                             vXF, &beta, vYF, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, fbuf);
            CUDA_CHECK(cudaDeviceSynchronize());
            double b0 = MPI_Wtime();
            for (int it = 0; it < iters; ++it)
                cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matF,
                             vXF, &beta, vYF, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, fbuf);
            CUDA_CHECK(cudaDeviceSynchronize());
            T1_ms = (MPI_Wtime() - b0) * 1000.0 / iters;
            if (fbuf) cudaFree(fbuf);
            cusparseDestroySpMat(matF);
            cusparseDestroyDnVec(vXF);
            cusparseDestroyDnVec(vYF);
        }

        print_metrics(MPI_COMM_WORLD, rank, P, "cuSPARSE KERNEL", nnz_total,
                      conv_ms, total_ms, comm_ms, compute_ms,
                      my_nnz, g.total_recv, mem_mb_cs, T1_ms, l2rel, maxabs, ok);

        if (d_buf) cudaFree(d_buf);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroy(handle);
    }

    if (rank == 0) {
        printf("\n# Done.\n");
        fflush(stdout);
    }

    /* ===================================================================== */
    /* Cleanup                                                                */
    /* ===================================================================== */
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_val);
    cudaFree(d_o_rp); cudaFree(d_o_ci); cudaFree(d_o_v);
    cudaFree(d_g_rp); cudaFree(d_g_ci); cudaFree(d_g_v);
    cudaFree(d_x); cudaFree(d_xloc); cudaFree(d_y);
    cudaFree(g.d_send_local); cudaFree(g.d_recv_global);
    cudaFree(g.d_send_buf);   cudaFree(g.d_recv_buf);
#if !CUDA_AWARE_MPI
    free(g.h_send_buf); free(g.h_recv_buf);
#endif
    free(g.send_counts); free(g.send_displs);
    free(g.recv_counts); free(g.recv_displs);
    free(g.reqs);
    free(recv_global); free(send_local);

    free(h_row_ptr); free(h_col_idx); free(h_val);
    free(loc_row); free(loc_col); free(loc_val); free(x_owned);
    free(h_yloc);

    if (rank == 0) {
        if (d_frp) cudaFree(d_frp);
        if (d_fci) cudaFree(d_fci);
        if (d_fv)  cudaFree(d_fv);
        if (d_fx)  cudaFree(d_fx);
        if (d_fy)  cudaFree(d_fy);
        free(fr_rp); free(fr_ci); free(fr_v); free(y_ref);
        free(y_all); free(y_dist); free(ycounts); free(ydispls);
        free(send_nnz); free(send_disp);
        free(pack_row); free(pack_col); free(pack_val);
        free(xs_cnt); free(xs_disp); free(pack_x);
        free(x_full);
        coo_free(&A);
    }

    MPI_Finalize();
    return 0;
}
