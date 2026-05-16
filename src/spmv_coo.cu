#include "../include/mtx_parser.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ==================== COO KERNELS ====================

// Kernel 1: COO-Standard (one thread per non-zero element)
// Simple approach: each thread processes one non-zero, uses atomicAdd for accumulation
__global__ void cooSpMV(int nnz, const int *row_idx, const int *col_idx, 
                               const FloatType *values, const FloatType *x, FloatType *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        int row = row_idx[idx];
        int col = col_idx[idx];
        FloatType val = values[idx];
        atomicAdd(&y[row], val * x[col]);
    }
}

// Kernel 2: COO-SortedSegmentedReduction (works on SORTED-by-row data)
// Uses warp-level segmented scan + reduction to minimize atomics
// REQUIRES: row_idx to be sorted in non-decreasing order
/*__global__ void cooSpMV_SortedSegmentedReduction(
        int nnz,
        const int       *__restrict__ row_idx,
        const int       *__restrict__ col_idx,
        const FloatType *__restrict__ values,
        const FloatType *__restrict__ x,
        FloatType        *__restrict__ y)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Each thread loads one non-zero (or becomes idle)
    FloatType val = 0;
    int       row = -1;
    if (tid < nnz) {
        row = row_idx[tid];
        val = values[tid] * x[col_idx[tid]];
    }

    // Intra-warp inclusive segmented scan:
    // Accumulate values for the same row using warp shuffles
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        FloatType nb_val = __shfl_up_sync(FULL_MASK, val, offset);
        int       nb_row = __shfl_up_sync(FULL_MASK, row, offset);
        if (lane >= offset && nb_row == row)
            val += nb_val;
    }

    // Flush: write accumulated value when:
    // (a) This is the last element, OR
    // (b) The next element belongs to a different row
    // Since data is SORTED by row, this uniquely identifies segment boundaries
    if (tid < nnz) {
        bool should_flush = false;
        
        if (tid + 1 >= nnz) {
            // Last element: always flush
            should_flush = true;
        } else {
            // Check next row (safe because data is sorted)
            int next_row = row_idx[tid + 1];
            if (next_row != row) {
                // Segment boundary: only last lane of segment flushes
                should_flush = true;
            }
        }
        
        if (should_flush) {
            atomicAdd(&y[row], val);
        }
    }
}*/
__global__ void cooSpMV_SortedSegmentedReduction(
        int nnz,
        const int       *__restrict__ row_idx,
        const int       *__restrict__ col_idx,
        const FloatType *__restrict__ values,
        const FloatType *__restrict__ x,
        FloatType        *__restrict__ y)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    FloatType val = 0.0f;
    int row = -1;

    if (tid < nnz) {
        row = row_idx[tid];
        val = values[tid] * x[col_idx[tid]];
    }

    // Warp-local segmented inclusive scan
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        FloatType nb_val =
            __shfl_up_sync(FULL_MASK, val, offset);

        int nb_row =
            __shfl_up_sync(FULL_MASK, row, offset);

        if (lane >= offset && nb_row == row)
            val += nb_val;
    }

    if (tid >= nnz)
        return;

    // Look only within warp
    int next_row =
        __shfl_down_sync(FULL_MASK, row, 1);

    bool end_segment =
            (lane == WARP_SIZE-1) ||   // force warp tail flush
            (tid == nnz-1) ||          // last global element
            (next_row != row);         // row ends in this warp

    if (end_segment)
        atomicAdd(&y[row], val);
}


// ==================== VALIDATION UTILITIES ====================

bool validateResult(const std::vector<FloatType>& gpu_result, 
                   const std::vector<FloatType>& cpu_result,
                   int size, const char *kernel_name) {

    FloatType tolerance = 1e-3;  // absolute floor
    FloatType rel_tolerance = 1e-4;  // relative tolerance
    int error_count = 0;
    FloatType max_error = 0.0f;

    
    for (int i = 0; i < size; i++) {
        FloatType diff = fabs(gpu_result[i] - cpu_result[i]);
        FloatType rel_error = diff / (fabs(cpu_result[i]) + 1e-10f);  // avoid div by zero
        if (diff > tolerance && rel_error > rel_tolerance) {
            error_count++;
            max_error = fmax(max_error, diff);
            if (error_count <= 5) {  // Print first 5 errors
                printf("  [%s] Error at index %d: GPU=%.6e, CPU=%.6e, diff=%.6e\n", 
                       kernel_name, i, gpu_result[i], cpu_result[i], diff);
            }
        }
    }
    
    if (error_count > 0) {
        printf("  [%s] FAILED: %d mismatches, max error: %.6e\n", kernel_name, error_count, max_error);
        return false;
    } else {
        printf("  [%s] PASSED\n", kernel_name);
        return true;
    }
}

// ==================== TIMING UTILITIES ====================

struct KernelStats {
    const char *name;
    double avg_time_ms;
    double gflops;
    bool validated;
};

KernelStats timeKernel_Standard(int m, int nnz,
                             const int *d_row_idx, const int *d_col_idx, const FloatType *d_values,
                             const FloatType *d_x, FloatType *d_y,
                             int warmup, int iterations, const char *name) {
    // Warm-up
    for (int i = 0; i < warmup; i++) {
        cooSpMV<<<(nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            nnz, d_row_idx, d_col_idx, d_values, d_x, d_y);
        // Clear y vector between iterations
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        cooSpMV<<<(nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            nnz, d_row_idx, d_col_idx, d_values, d_x, d_y);
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_ms = 0; //check comment
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    
    double avg_time = time_ms / iterations;
    double gflops = (2.0 * nnz) / (avg_time * 1e-3) / 1e9;  // 2*nnz flops per SpMV
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    KernelStats stats;
    stats.name = name;
    stats.avg_time_ms = avg_time;
    stats.gflops = gflops;
    stats.validated = false;
    return stats;
}


KernelStats timeKernel_SortedSegmentedReduction(int m, int nnz,
                                          const int *d_row_idx, const int *d_col_idx, const FloatType *d_values,
                                          const FloatType *d_x, FloatType *d_y,
                                          int warmup, int iterations, const char *name) {
    // Warm-up
    for (int i = 0; i < warmup; i++) {
        cooSpMV_SortedSegmentedReduction<<<(nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            nnz, d_row_idx, d_col_idx, d_values, d_x, d_y);
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        cooSpMV_SortedSegmentedReduction<<<(nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            nnz, d_row_idx, d_col_idx, d_values, d_x, d_y);
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    
    double avg_time = time_ms / iterations;
    double gflops = (2.0 * nnz) / (avg_time * 1e-3) / 1e9;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    KernelStats stats;
    stats.name = name;
    stats.avg_time_ms = avg_time;
    stats.gflops = gflops;
    stats.validated = false;
    return stats;
}

// ==================== MAIN ====================

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <matrix.mtx> <warmup_cycles> <iterations>\n", argv[0]);
        return 1;
    }

    const char *mtx_file = argv[1];
    int warmup = atoi(argv[2]);
    int iterations = atoi(argv[3]);

    if (warmup <= 0 || iterations <= 0) {
        fprintf(stderr, "Error: warmup and iterations must be positive\n");
        return 1;
    }

    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║           COO Format SpMV Benchmark                       ║\n");
    printf("║     Standard vs Sorted-Segmented Reduction                ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    // Read matrix
    printf("\nLoading matrix from: %s\n", mtx_file);
    COOMatrix coo = readMatrixMarket(mtx_file);
    
    // ===== SORT BY ROW FOR KERNEL 2 =====
    printf("\nSorting COO data by row index for Kernel 2...\n");
    std::vector<int> idx(coo.nnz);
    for (int i = 0; i < coo.nnz; i++) idx[i] = i;
    
    std::sort(idx.begin(), idx.end(), [&](int i, int j) {
        return coo.row_indices[i] < coo.row_indices[j];
    });
    
    // Create sorted copies
    std::vector<int> sorted_row(coo.nnz), sorted_col(coo.nnz);
    std::vector<FloatType> sorted_val(coo.nnz);
    for (int i = 0; i < coo.nnz; i++) {
        sorted_row[i] = coo.row_indices[idx[i]];
        sorted_col[i] = coo.col_indices[idx[i]];
        sorted_val[i] = coo.values[idx[i]];
    }

    // Allocate device memory for COO format
    int *d_row_idx, *d_col_idx;
    int *d_sorted_row, *d_sorted_col;
    FloatType *d_values, *d_sorted_val, *d_x, *d_y;
    
    CUDA_CHECK(cudaMalloc(&d_row_idx, coo.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, coo.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, coo.nnz * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_sorted_row, coo.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_col, coo.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_val, coo.nnz * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_x, coo.n * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_y, coo.m * sizeof(FloatType)));

    // Copy unsorted data for Kernel 1
    CUDA_CHECK(cudaMemcpy(d_row_idx, coo.row_indices.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, coo.col_indices.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, coo.values.data(), coo.nnz * sizeof(FloatType), cudaMemcpyHostToDevice));

    // Copy sorted data for Kernel 2
    CUDA_CHECK(cudaMemcpy(d_sorted_row, sorted_row.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sorted_col, sorted_col.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sorted_val, sorted_val.data(), coo.nnz * sizeof(FloatType), cudaMemcpyHostToDevice));

    // Generate random vector
    printf("Generating random vector (seed=42)...\n");
    generateRandomVector(d_x, coo.n, 42);

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("COO SpMV Details\n");
    printf("Grid size: %d, Block size: %d\n", (coo.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE);
    printf("Warmup cycles: %d, Iterations: %d\n", warmup, iterations);
    printf("Note: Kernel 2 uses SORTED-by-row data\n");

    // Benchmark kernels
    std::vector<KernelStats> results;
    
    results.push_back(timeKernel_Standard(
        coo.m, coo.nnz,
        d_row_idx, d_col_idx, d_values, d_x, d_y,
        warmup, iterations, "COO-Standard (unsorted)"
    ));

    results.push_back(timeKernel_SortedSegmentedReduction(
        coo.m, coo.nnz,
        d_sorted_row, d_sorted_col, d_sorted_val, d_x, d_y,
        warmup, iterations, "COO-SortedSegmentedReduction"
    ));

    // Print results
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("Benchmark Results Summary\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    for (const auto &stat : results) {
        printf("%-40s: %8.4f ms | %10.2f GFLOP/s\n", stat.name, stat.avg_time_ms, stat.gflops);
    }

    // Find best kernel
    int best_idx = 0;
    double best_gflops = results[0].gflops;
    for (int i = 1; i < (int)results.size(); i++) {
        if (results[i].gflops > best_gflops) {
            best_gflops = results[i].gflops;
            best_idx = i;
        }
    }

    printf("\nBest performing kernel: %s (%.2f GFLOP/s)\n\n", results[best_idx].name, best_gflops);

    // Calculate speedups relative to standard
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Speedup relative to COO-Standard:\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    for (int i = 1; i < (int)results.size(); i++) {
        double speedup = results[0].avg_time_ms / results[i].avg_time_ms;
        double improvement = (speedup - 1.0) * 100.0;
        printf("%-40s: %.2fx speedup (%.1f%% improvement)\n", 
               results[i].name, speedup, improvement);
    }

    // ==================== CPU BASELINE VALIDATION ====================
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("Validation Against CPU Baseline (OpenMP)\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    // Allocate CPU vectors
    std::vector<FloatType> h_x(coo.n);
    std::vector<FloatType> h_y_cpu(coo.m, 0.0f);

    // Generate same random vector on CPU
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, coo.n * sizeof(FloatType), cudaMemcpyDeviceToHost));

    // Run CPU baseline (once)
    printf("Running CPU baseline with OpenMP...\n");

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < coo.nnz; i++) {
        int row = coo.row_indices[i];
        int col = coo.col_indices[i];
        #pragma omp atomic
        h_y_cpu[row] += coo.values[i] * h_x[col];
    }


    // Validate each kernel against CPU baseline
    printf("Comparing GPU kernels against CPU baseline:\n");
    printf("───────────────────────────────────────────────────────────\n");

    for (size_t k = 0; k < results.size(); k++) {
        // Allocate GPU result vector
        std::vector<FloatType> h_y_gpu(coo.m);

        // Run kernel one final time and get result
        CUDA_CHECK(cudaMemset(d_y, 0, coo.m * sizeof(FloatType)));
        if (k == 0) {
            cooSpMV<<<(coo.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                coo.nnz, d_row_idx, d_col_idx, d_values, d_x, d_y);
        } else {
            cooSpMV_SortedSegmentedReduction<<<(coo.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                coo.nnz, d_sorted_row, d_sorted_col, d_sorted_val, d_x, d_y);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy GPU result to host
        CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, coo.m * sizeof(FloatType), cudaMemcpyDeviceToHost));

        // Validate
        results[k].validated = validateResult(h_y_gpu, h_y_cpu, coo.m, results[k].name);

        // Clear d_y for next kernel
        CUDA_CHECK(cudaMemset(d_y, 0, coo.m * sizeof(FloatType)));
    }
    
    printf("═══════════════════════════════════════════════════════════\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_row_idx));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_sorted_row));
    CUDA_CHECK(cudaFree(d_sorted_col));
    CUDA_CHECK(cudaFree(d_sorted_val));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    printf("COO benchmark completed successfully!\n\n");
    return 0;
}
