#include "../include/mtx_parser.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

// ============================================================
//  Tuning knobs  (match paper: THREADS=512, R=2)
// ============================================================
#define BLOCK_SIZE   512          // THREADS per workgroup
#define WARP_SIZE    32
#define R_FLAT       2            // NNZ processed per thread per round in flat
#define R_LINE       2            // same for line-enhance
// NNZ processed by one workgroup per round = R * THREADS
#define STRIDE_FLAT  (R_FLAT * BLOCK_SIZE)   // = 1024
#define STRIDE_LINE  (R_LINE * BLOCK_SIZE)   // = 1024

// ============================================================
//  Vector size V for the reduction step inside line-enhance.
//  Paper: V=1 for short-row matrices (avg nnz/row < 24),
//         V=4 for longer rows.  We expose both via a template.
// ============================================================

// ============================================================
//  CSR-VECTOR – Classic warp-per-row SpMV
//  Each warp (32 threads) processes one row of the matrix.
//  Threads within a warp cooperatively reduce the dot product
//  using warp shuffle instructions.
// ============================================================
__global__ void csrSpMV_Vector(
        int m,
        const int      *row_ptr,
        const int      *col_idx,
        const FloatType *values,
        const FloatType *x,
        FloatType       *y)
{
    // Global warp id → one warp per row
    int warp_id    = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane       = threadIdx.x % WARP_SIZE;   // lane inside the warp [0,31]

    if (warp_id >= m) return;

    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];

    // Each lane accumulates a partial sum strided by WARP_SIZE
    FloatType sum = 0.0f;
    for (int j = row_start + lane; j < row_end; j += WARP_SIZE)
        sum += values[j] * x[col_idx[j]];

    // Warp-level reduction via shuffle down
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    // Lane 0 writes the result
    if (lane == 0)
        y[warp_id] = sum;
}

// ============================================================
//  FLAT – Algorithm 2 (preprocessing) + Algorithm 3 (SpMV)
//  NNZ-splitting: each workgroup processes STRIDE_FLAT non-zeros.
// ============================================================

// ------ Preprocessing kernel (Algorithm 2) ------
// Generates the break-point array bp[0..WGS] where
//   bp[i] = row-id of the first non-zero assigned to workgroup i.
__global__ void flat_preprocess(
        int m, int nnz,
        const int *row_ptr,        // size m+1
        int       *bp,             // size WGS+1  (output)
        int        WGS,            // number of workgroups in the SpMV kernel
        int        STRIDE)         // non-zeros per workgroup  (= STRIDE_FLAT)
{
    // thread 0 initialises bp[0] (paper line 1)
    if (blockIdx.x == 0 && threadIdx.x == 0)
        bp[0] = 0;

    // Each thread iterates over a subset of rows (paper lines 2-9)
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = g_tid; i < m; i += total_threads) {
        int cur_bp  = row_ptr[i]     / STRIDE;   // workgroup id for row i start
        int next_bp = row_ptr[i + 1] / STRIDE;   // workgroup id for row i+1 start

        if (cur_bp != next_bp) {
            // Fill bp[cur_bp+1 .. next_bp] with row i   (paper lines 6-7)
            for (int j = cur_bp + 1; j <= next_bp && j <= WGS; j++)
                bp[j] = i;

            // Special case: row i+1 starts exactly at a workgroup boundary
            // -> the first nnz of workgroup next_bp belongs to row i+1, not i
            // so we add 1 to bp[next_bp]  (paper lines 8-9)
            if ((row_ptr[i + 1] % STRIDE) == 0 && next_bp <= WGS)
                atomicAdd(&bp[next_bp], 1);
        }
    }
}

// ------ Core SpMV flat kernel (Algorithm 3) ------
// Each workgroup:
//   1. Loads R*THREADS multiplications into shared memory (LDS).
//   2. Reads bp to find its row range [reduce_row_start, reduce_row_end).
//   3. Each thread reduces one (or more) rows directly from shared memory.
//   4. atomicAdd results back to y (a row may span multiple workgroups).
__global__ void csrSpMV_Flat(
        int m, int nnz,
        const int      *row_ptr,
        const int      *col_idx,
        const FloatType *values,
        const FloatType *x,
        FloatType       *y,
        const int       *bp,       // break-point array
        int              WGS,
        int              STRIDE)   // NNZ per workgroup = R * THREADS
{
    // Shared memory buffer: STRIDE = R * BLOCK_SIZE entries
    __shared__ FloatType smem[STRIDE_FLAT];

    int wg_id   = blockIdx.x;                   // workgroup id
    int tid     = threadIdx.x;                  // thread id in workgroup
    int THREADS = blockDim.x;                   // = BLOCK_SIZE

    // Global NNZ range for this workgroup
    int wg_nnz_start = wg_id * STRIDE;
    int wg_nnz_end   = min(wg_nnz_start + STRIDE, nnz);
    if (wg_nnz_start >= nnz) return;            // workgroup has nothing to do

    // ---- Step 1: data loading + multiplication → LDS (paper lines 7-11) ----
    // Loop unrolling: paper suggests R iterations
    for (int r = 0; r < R_FLAT; r++) {
        int idx = wg_nnz_start + r * THREADS + tid;
        int smem_idx = r * THREADS + tid;
        FloatType tmp = 0.0f;
        if (idx < wg_nnz_end)
            tmp = values[idx] * x[col_idx[idx]];
        smem[smem_idx] = tmp;
    }
    __syncthreads();

    // ---- Step 2: calculate reduction row range (paper lines 13-17) ----
    int bp_idx = wg_id;   // break_point_idx for this workgroup

    int reduce_row_start = min(bp[bp_idx],     m);
    int reduce_row_end   = (bp_idx + 1 <= WGS) ? min(bp[bp_idx + 1], m) : m;
    if (reduce_row_end == 0) reduce_row_end = m;   // paper line 15

    // Paper lines 16-17: if next workgroup's first nnz is exactly at a
    // row boundary, or the entire workgroup belongs to one row, expand end.
    if (wg_nnz_end < nnz) {
        if ((row_ptr[reduce_row_end] % STRIDE) == 0 ||
            reduce_row_start == reduce_row_end)
            reduce_row_end = min(reduce_row_end + 1, m);
    } else {
        // Last workgroup always runs to end
        reduce_row_end = m;
    }

    // bp_nnz_id: the global NNZ index where this workgroup starts (paper line 19)
    int bp_nnz_id = bp_idx * STRIDE;

    // ---- Step 3: reduction + atomic store (paper lines 20-27) ----
    // Each thread is responsible for one or more rows in [reduce_row_start, reduce_row_end)
    int reduce_row_id = reduce_row_start + tid;   // paper line 18

    while (reduce_row_id < reduce_row_end) {
        FloatType sum = 0.0f;

        // Range of entries in smem that belong to this row
        int rstart = max(0,      row_ptr[reduce_row_id]     - bp_nnz_id);
        int rend   = min(STRIDE, row_ptr[reduce_row_id + 1] - bp_nnz_id);

        for (int i = rstart; i < rend; i++)
            sum += smem[i];

        // Atomic because a row may span multiple workgroups
        atomicAdd(&y[reduce_row_id], sum);

        reduce_row_id += THREADS;   // paper line 27
    }
}

// ============================================================
//  LINE-ENHANCE – Algorithm 4 (SpMV) + Algorithm 5 (VecReduce)
//  Hybrid row+NNZ splitting: each workgroup owns N entire rows;
//  inside the workgroup NNZs are split equally across threads.
//  No preprocessing, no atomic stores to y.
//
//  Template parameter V: number of threads per "vector" used for
//  the intra-workgroup reduction of one row.
//  V=1  → one thread per row  (good for short rows)
//  V=4  → four threads per row (good for longer rows)
// ============================================================
template <int V, int N_ROWS>
__global__ void csrSpMV_LineEnhance(
        int m,
        const int      *row_ptr,
        const int      *col_idx,
        const FloatType *values,
        const FloatType *x,
        FloatType       *y)
{
    // Shared memory: STRIDE_LINE entries (Algorithm 4, line 6)
    __shared__ FloatType smem[STRIDE_LINE];

    int wg_id   = blockIdx.x;
    int tid     = threadIdx.x;
    int THREADS = blockDim.x;   // = BLOCK_SIZE

    // ---- Task partitioning (Algorithm 4, lines 1-5) ----
    int wg_row_begin = wg_id * N_ROWS;
    int wg_row_end   = min(wg_row_begin + N_ROWS, m);
    if (wg_row_begin >= m) return;

    int wg_nnz_start = row_ptr[wg_row_begin];
    int wg_nnz_end   = row_ptr[wg_row_end];

    // Total rounds: ceil(total_nnz / (R * THREADS))
    int total_nnz = wg_nnz_end - wg_nnz_start;
    int rounds    = (total_nnz + STRIDE_LINE - 1) / STRIDE_LINE;

    // Per-vector accumulator, kept across rounds (Algorithm 4, line 6)
    // Each thread belongs to exactly one vector: vec_id = tid / V
    // Number of vectors in the workgroup = THREADS / V
    // Vector i is responsible for row  wg_row_begin + vec_id
    int vec_id     = tid / V;           // which vector this thread belongs to
    int tid_in_vec = tid % V;           // lane inside the vector

    // Accumulator for this thread's contribution to its row
    FloatType sum = 0.0f;

    // ---- Rounds loop (Algorithm 4, lines 7-18) ----
    for (int r = 0; r < rounds; r++) {

        int round_start = wg_nnz_start + r * STRIDE_LINE;
        int round_end   = min(round_start + STRIDE_LINE, wg_nnz_end);

        __syncthreads();   // Algorithm 4, line 10 (guard LDS from previous round)

        // ---- Multiplication → shared memory (Algorithm 4, lines 11-16) ----
        // All threads cooperate to load R*THREADS entries in a coalesced fashion.
        int i = round_start + tid;
        for (int k = 0; k < R_LINE; k++) {
            int smem_idx = k * THREADS + tid;
            FloatType tmp = 0.0f;
            if (i < round_end)
                tmp = values[i] * x[col_idx[i]];
            smem[smem_idx] = tmp;
            i += THREADS;
        }

        __syncthreads();   // Algorithm 4, line 17: wait for LDS write

        // ---- VecReduce: Algorithm 5 ----
        // Each vector reduces the entries in smem that belong to its row
        // intersected with the current round.

        int my_row = wg_row_begin + vec_id;
        if (my_row < wg_row_end) {
            // vec_range = [row_ptr[my_row], row_ptr[my_row+1])  (paper: I_begin, I_end)
            int I_begin = row_ptr[my_row];
            int I_end   = row_ptr[my_row + 1];

            // reduce_range = vec_range ∩ round_range  (Algorithm 5, lines 3-6)
            int R_begin = max(I_begin, round_start);
            int R_end   = min(I_end,   round_end);

            // Parallel reduction across V threads in the vector (Algorithm 5, line 7)
            for (int j = R_begin + tid_in_vec; j < R_end; j += V) {
                sum += smem[j - round_start];
            }
        }
        // sum accumulates across rounds (Algorithm 4, line 18: sum += VecReduce(...))
    }

    // ---- Store results (Algorithm 4, lines 19-22) ----
    // Collect partial sums within the vector using warp shuffle prefix sum.
    // Paper uses VecPrefixSum when V > 1.
    if (V > 1) {
        // Inclusive prefix sum (scan) inside the vector using shuffle.
        // We want the first thread (tid_in_vec == 0) to hold the total.
        for (int offset = 1; offset < V; offset *= 2) {
            FloatType tmp = __shfl_up_sync(0xFFFFFFFF, sum, offset);
            if (tid_in_vec >= offset) sum += tmp;
        }
        // After prefix-sum, thread with tid_in_vec == V-1 holds the total.
        // Broadcast to tid_in_vec == 0 for storing.
        sum = __shfl_sync(0xFFFFFFFF, sum, (vec_id * V) + (V - 1));
    }

    // Only the first thread of each vector stores the result (Algorithm 4, line 21)
    int my_row = wg_row_begin + vec_id;
    if (tid_in_vec == 0 && my_row < wg_row_end) {
        y[my_row] = sum;   // α=1, β=0 simplification (no α·Ax + β·y scaling here)
    }
}

// ==================== VALIDATION ====================

bool validateResult(const std::vector<FloatType>& gpu_result, 
                   const std::vector<FloatType>& cpu_result,
                   int size, FloatType tolerance = 1e-4) {
    for (int i = 0; i < size; i++) {
        FloatType error = fabs(gpu_result[i] - cpu_result[i]);
        FloatType rel_error = error / (fabs(cpu_result[i]) + 1e-10);
        if (error > tolerance && rel_error > tolerance) {
            printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f (error=%.6e, rel_error=%.6e)\n",
                   i, gpu_result[i], cpu_result[i], error, rel_error);
            return false;
        }
    }
    return true;
}

// ============================================================
//  Timing utilities
// ============================================================
struct KernelStats {
    const char *name;
    double avg_time_ms;
    double gflops;
};

// ---- CSR-Vector benchmark wrapper ----
KernelStats timeKernel_Vector(
        int m, int nnz,
        const int *d_row_ptr, const int *d_col_idx,
        const FloatType *d_values, const FloatType *d_x, FloatType *d_y,
        int warmup, int iterations)
{
    // One warp per row → total warps = m, threads = m * WARP_SIZE
    // Pack into blocks of BLOCK_SIZE threads (= BLOCK_SIZE/WARP_SIZE warps/block)
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    int WGS = (m + warps_per_block - 1) / warps_per_block;

    // Warm-up
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
        csrSpMV_Vector<<<WGS, BLOCK_SIZE>>>(
            m, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
        csrSpMV_Vector<<<WGS, BLOCK_SIZE>>>(
            m, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    double avg_ms = elapsed_ms / iterations;
    double gflops = (2.0 * nnz) / (avg_ms * 1e-3) / 1e9;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    KernelStats s;
    s.name        = "CSR-Vector (warp-per-row)";
    s.avg_time_ms = avg_ms;
    s.gflops      = gflops;
    return s;
}

// ---- Flat benchmark wrapper ----
KernelStats timeKernel_Flat(
        int m, int nnz,
        const int *d_row_ptr, const int *d_col_idx,
        const FloatType *d_values, const FloatType *d_x, FloatType *d_y,
        int warmup, int iterations)
{
    // --- Preprocessing ---
    int WGS    = (nnz + STRIDE_FLAT - 1) / STRIDE_FLAT;   // number of workgroups
    int *d_bp;
    CUDA_CHECK(cudaMalloc(&d_bp, (WGS + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bp, 0, (WGS + 1) * sizeof(int)));

    // Launch preprocessing kernel
    int pre_threads = BLOCK_SIZE;
    int pre_blocks  = (m + pre_threads - 1) / pre_threads;
    flat_preprocess<<<pre_blocks, pre_threads>>>(m, nnz, d_row_ptr, d_bp, WGS, STRIDE_FLAT);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warm-up
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
        csrSpMV_Flat<<<WGS, BLOCK_SIZE>>>(
            m, nnz, d_row_ptr, d_col_idx, d_values, d_x, d_y,
            d_bp, WGS, STRIDE_FLAT);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing (excludes preprocessing, matches paper's "kernel time")
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
        csrSpMV_Flat<<<WGS, BLOCK_SIZE>>>(
            m, nnz, d_row_ptr, d_col_idx, d_values, d_x, d_y,
            d_bp, WGS, STRIDE_FLAT);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    double avg_ms   = elapsed_ms / iterations;
    double gflops   = (2.0 * nnz) / (avg_ms * 1e-3) / 1e9;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_bp));

    KernelStats s;
    s.name       = "Flat (Chu 2023)";
    s.avg_time_ms = avg_ms;
    s.gflops      = gflops;
    return s;
}

// ---- Line-enhance benchmark wrapper ----
// Chooses V and N_ROWS based on avg nnz/row (paper Section 3.3.2)
KernelStats timeKernel_LineEnhance(
        int m, int nnz,
        const int *d_row_ptr, const int *d_col_idx,
        const FloatType *d_values, const FloatType *d_x, FloatType *d_y,
        int warmup, int iterations)
{
    double avg_nnz = (double)nnz / m;

    // Paper: if avg nnz/row > 24 → N=64, V=4; else → N=128, V=1
    // Constraint: THREADS/V <= N  →  512/V <= N  (always satisfied below)
    int N_ROWS_VAL, V_VAL;
    if (avg_nnz > 24.0) {
        N_ROWS_VAL = 64;
        V_VAL      = 4;
    } else {
        N_ROWS_VAL = 128;
        V_VAL      = 1;
    }

    int WGS = (m + N_ROWS_VAL - 1) / N_ROWS_VAL;

    // Warm-up
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
        if (V_VAL == 1) {
            csrSpMV_LineEnhance<1, 128><<<WGS, BLOCK_SIZE>>>(
                m, d_row_ptr, d_col_idx, d_values, d_x, d_y);
        } else {
            csrSpMV_LineEnhance<4, 64><<<WGS, BLOCK_SIZE>>>(
                m, d_row_ptr, d_col_idx, d_values, d_x, d_y);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
        if (V_VAL == 1) {
            csrSpMV_LineEnhance<1, 128><<<WGS, BLOCK_SIZE>>>(
                m, d_row_ptr, d_col_idx, d_values, d_x, d_y);
        } else {
            csrSpMV_LineEnhance<4, 64><<<WGS, BLOCK_SIZE>>>(
                m, d_row_ptr, d_col_idx, d_values, d_x, d_y);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    double avg_ms = elapsed_ms / iterations;
    double gflops = (2.0 * nnz) / (avg_ms * 1e-3) / 1e9;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    KernelStats s;
    s.name        = "Line-Enhance (Chu 2023)";
    s.avg_time_ms = avg_ms;
    s.gflops      = gflops;
    return s;
}

// ============================================================
//  main
// ============================================================
int main(int argc, char *argv[])
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <matrix.mtx> <warmup_cycles> <iterations>\n", argv[0]);
        return 1;
    }

    const char *mtx_file = argv[1];
    int warmup     = atoi(argv[2]);
    int iterations = atoi(argv[3]);

    if (warmup <= 0 || iterations <= 0) {
        fprintf(stderr, "Error: warmup and iterations must be positive\n");
        return 1;
    }


    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║    Chu et al. 2023 – Flat & Line-Enhance SpMV Benchmark   ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("Loading matrix from: %s\n", mtx_file);
    COOMatrix coo = readMatrixMarket(mtx_file);
    

    printf("\nConverting to CSR format...\n");
    CSRMatrix csr = cooToCSR(coo);
    allocateCSRMatrixGPU(csr);

    // Row-length statistics
    std::vector<int> h_row_ptr(coo.m + 1);
    CUDA_CHECK(cudaMemcpy(h_row_ptr.data(), csr.d_row_ptr,
                          (coo.m + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    int max_len = 0;
    for (int i = 0; i < coo.m; i++)
        max_len = std::max(max_len, h_row_ptr[i + 1] - h_row_ptr[i]);
    double avg_nnz = (double)coo.nnz / coo.m;
    printf("\nRow statistics: avg nnz/row = %.2f, max nnz/row = %d\n",
           avg_nnz, max_len);

    // Allocate vectors
    FloatType *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, coo.n * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_y, coo.m * sizeof(FloatType)));
    generateRandomVector(d_x, coo.n, 42);

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("Parameters: BLOCK_SIZE=%d, R=%d, STRIDE=%d\n",
           BLOCK_SIZE, R_FLAT, STRIDE_FLAT);
    printf("Flat workgroups (WGS) = %d\n",
           (coo.nnz + STRIDE_FLAT - 1) / STRIDE_FLAT);
    printf("Line-Enhance config: avg_nnz=%.1f → %s\n",
           avg_nnz, avg_nnz > 24.0 ? "N=64, V=4" : "N=128, V=1");
    printf("Warmup: %d, Iterations: %d\n", warmup, iterations);
    printf("═══════════════════════════════════════════════════════════\n\n");

    std::vector<KernelStats> results;

    printf("Benchmarking CSR-Vector kernel...\n");
    results.push_back(timeKernel_Vector(
        coo.m, coo.nnz,
        csr.d_row_ptr, csr.d_col_idx, csr.d_values, d_x, d_y,
        warmup, iterations));

    printf("Benchmarking Flat kernel...\n");
    results.push_back(timeKernel_Flat(
        coo.m, coo.nnz,
        csr.d_row_ptr, csr.d_col_idx, csr.d_values, d_x, d_y,
        warmup, iterations));

    printf("Benchmarking Line-Enhance kernel...\n");
    results.push_back(timeKernel_LineEnhance(
        coo.m, coo.nnz,
        csr.d_row_ptr, csr.d_col_idx, csr.d_values, d_x, d_y,
        warmup, iterations));

    // Results
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("Benchmark Results\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    for (const auto &s : results)
        printf("%-30s: %8.4f ms | %10.2f GFLOP/s\n",
               s.name, s.avg_time_ms, s.gflops);

    int best = 0;
    for (int i = 1; i < (int)results.size(); i++)
        if (results[i].gflops > results[best].gflops) best = i;
    printf("\n✓ Best: %s (%.2f GFLOP/s)\n\n",
           results[best].name, results[best].gflops);
    
     // ==================== VALIDATION ====================
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("Validation Against CPU Baseline (OpenMP)\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    // Allocate CPU vectors
    std::vector<FloatType> h_x(coo.n);
    std::vector<FloatType> h_y_gpu(coo.m);
    std::vector<FloatType> h_y_cpu(coo.m, 0.0f);

    // Generate same random vector on CPU
    srand(42);
    for (int i = 0; i < coo.n; i++) {
        h_x[i] = (FloatType)rand() / RAND_MAX;
    }

    // Copy GPU result
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, coo.m * sizeof(FloatType), cudaMemcpyDeviceToHost));

    // Run CPU baseline
    printf("Running CPU baseline with OpenMP...\n");
    spmvCPU_CSR(coo.m, coo.n, csr.row_ptr.data(), csr.col_idx.data(), 
                csr.values.data(), h_x.data(), h_y_cpu.data());

    // Validate results
    printf("Comparing GPU and CPU results...\n");
    bool valid = validateResult(h_y_gpu, h_y_cpu, coo.m);

    if (valid) {
        printf("✓ Validation PASSED: GPU results match CPU baseline\n\n");
    } else {
        printf("✗ Validation FAILED: GPU results differ from CPU baseline\n\n");
    }
    

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    freeCSRMatrix(csr);

    printf("✓ Benchmark completed.\n\n");
    return 0;
}
