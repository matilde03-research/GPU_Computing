#include "../include/mtx_parser.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

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

// Kernel 2: COO-SegmentedReduction (improved with segmented warp operations)
// Groups non-zeros by row and performs efficient segmented reduction within warps
__global__ void cooSpMV_SegmentedReduction(int nnz, const int *row_idx, const int *col_idx, 
                                           const FloatType *values, const FloatType *x, FloatType *y) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int elements_per_warp = (nnz + gridDim.x * blockDim.x / WARP_SIZE - 1) / (gridDim.x * blockDim.x / WARP_SIZE);
    
    int start_idx = warp_id * elements_per_warp;
    int end_idx = min((warp_id + 1) * elements_per_warp, nnz);
    
    // Process assigned elements
    for (int idx = start_idx + lane; idx < end_idx; idx += WARP_SIZE) {
        int row = row_idx[idx];
        int col = col_idx[idx];
        FloatType val = values[idx];
        
        // Accumulate using atomicAdd
        atomicAdd(&y[row], val * x[col]);
    }
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
    
    float time_ms = 0;
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


KernelStats timeKernel_SegmentedReduction(int m, int nnz,
                                          const int *d_row_idx, const int *d_col_idx, const FloatType *d_values,
                                          const FloatType *d_x, FloatType *d_y,
                                          int warmup, int iterations, const char *name) {
    // Warm-up
    for (int i = 0; i < warmup; i++) {
        cooSpMV_SegmentedReduction<<<(nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
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
        cooSpMV_SegmentedReduction<<<(nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
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
    printf("║     Standard vs Segmented Reduction                       ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    // Read matrix
    printf("\nLoading matrix from: %s\n", mtx_file);
    COOMatrix coo = readMatrixMarket(mtx_file);
    

    // Allocate device memory for COO format
    int *d_row_idx, *d_col_idx;
    FloatType *d_values, *d_x, *d_y;
    
    CUDA_CHECK(cudaMalloc(&d_row_idx, coo.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, coo.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, coo.nnz * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_x, coo.n * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_y, coo.m * sizeof(FloatType)));

    // Copy COO data to device
    CUDA_CHECK(cudaMemcpy(d_row_idx, coo.row_indices.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, coo.col_indices.data(), coo.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, coo.values.data(), coo.nnz * sizeof(FloatType), cudaMemcpyHostToDevice));

    // Generate random vector
    printf("Generating random vector (seed=42)...\n");
    generateRandomVector(d_x, coo.n, 42);

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("COO SpMV Benchmark Results\n");
    printf("Matrix: %d × %d, NNZ: %d\n", coo.m, coo.n, coo.nnz);
    printf("Grid size: %d, Block size: %d\n", (coo.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE);
    printf("Warmup cycles: %d, Iterations: %d\n", warmup, iterations);
    

    // Benchmark kernels
    std::vector<KernelStats> results;
    
    //printf("Benchmarking Kernel 1: COO-Standard...\n");
    results.push_back(timeKernel_Standard(
        coo.m, coo.nnz,
        d_row_idx, d_col_idx, d_values, d_x, d_y,
        warmup, iterations, "COO-Standard"
    ));

    //printf("Benchmarking Kernel 2: COO-SegmentedReduction...\n");
    results.push_back(timeKernel_SegmentedReduction(
        coo.m, coo.nnz,
        d_row_idx, d_col_idx, d_values, d_x, d_y,
        warmup, iterations, "COO-SegmentedReduction"
    ));

    // Print results
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("Benchmark Results Summary\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    for (const auto &stat : results) {
        printf("%-30s: %8.4f ms | %10.2f GFLOP/s\n", stat.name, stat.avg_time_ms, stat.gflops);
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

    printf("\n Best performing kernel: %s (%.2f GFLOP/s)\n\n", results[best_idx].name, best_gflops);

    // Calculate speedups relative to standard
    printf("═══════════════════════════════════════════════════════════\n");
    printf("Speedup relative to COO-Standard:\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    for (int i = 1; i < (int)results.size(); i++) {
        double speedup = results[0].avg_time_ms / results[i].avg_time_ms;
        double improvement = (speedup - 1.0) * 100.0;
        printf("%-30s: %.2fx speedup (%.1f%% improvement)\n", 
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
    //printf("Copying GPU x vector to host for CPU validation...\n");
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
            cooSpMV_SegmentedReduction<<<(coo.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                coo.nnz, d_row_idx, d_col_idx, d_values, d_x, d_y);
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
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    printf("COO benchmark completed successfully!\n\n");
    return 0;
}
