#include "../include/mtx_parser.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ==================== CSR KERNELS ====================

// Kernel 1: CSR-Vector (one thread per row)
__global__ void csrSpMV_Vector(int m, const int *row_ptr, const int *col_idx, 
                                const FloatType *values, const FloatType *x, FloatType *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        FloatType sum = 0.0f;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

// Kernel 2: CSR-Flat (warp per row - multiple threads per row)
__global__ void csrSpMV_Flat(int m, const int *row_ptr, const int *col_idx, 
                              const FloatType *values, const FloatType *x, FloatType *y) {
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    if (row < m) {
        FloatType sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        // Process elements in the row using warp threads
        for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
            sum += values[j] * x[col_idx[j]];
        }
        
        // Warp reduction using __shfl_down_sync
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        
        if (lane == 0) {
            y[row] = sum;
        }
    }
}

// Kernel 3: CSR-Line-Enhance (adaptive parallelization)
// Based on Chu et al. "Efficient Algorithm Design of Optimizing SpMV on GPU"
__global__ void csrSpMV_LineEnhance(int m, const int *row_ptr, const int *col_idx, 
                                     const FloatType *values, const FloatType *x, FloatType *y) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane = thread_id % WARP_SIZE;
    
    if (warp_id < m) {
        FloatType sum = 0.0f;
        int row_start = row_ptr[warp_id];
        int row_end = row_ptr[warp_id + 1];
        int row_len = row_end - row_start;
        
        // Adaptive strategy based on row length
        if (row_len <= WARP_SIZE) {
            // Short row: use all threads in warp
            for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
                sum += values[j] * x[col_idx[j]];
            }
            
            // Reduction
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            
            if (lane == 0) {
                y[warp_id] = sum;
            }
        } else {
            // Long row: each thread computes its portion
            for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
                sum += values[j] * x[col_idx[j]];
            }
            
            // Still need to reduce within warp
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            
            if (lane == 0) {
                y[warp_id] = sum;
            }
        }
    }
}

// ==================== TIMING UTILITIES ====================

struct KernelStats {
    const char *name;
    double avg_time_ms;
    double gflops;
};

KernelStats timeKernel(void (*kernel)(int, const int*, const int*, const FloatType*, const FloatType*, FloatType*),
                       int m, int nnz,
                       const int *d_row_ptr, const int *d_col_idx, const FloatType *d_values,
                       const FloatType *d_x, FloatType *d_y,
                       int warmup, int iterations, const char *name) {
    // Warm-up
    for (int i = 0; i < warmup; i++) {
        kernel<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(m, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(m, d_row_ptr, d_col_idx, d_values, d_x, d_y);
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

    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║           CSR Format SpMV Benchmark                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    // Read matrix
    printf("Loading matrix from: %s\n", mtx_file);
    COOMatrix coo = readMatrixMarket(mtx_file);
    printMatrixStats(coo);

    // Convert to CSR (with timing)
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Converting to CSR format...\n");
    CSRMatrix csr = cooToCSR(coo);

    // Allocate vectors on device
    FloatType *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, coo.n * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_y, coo.m * sizeof(FloatType)));

    // Generate random vector
    printf("Generating random vector (seed=42)...\n");
    generateRandomVector(d_x, coo.n, 42);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("CSR SpMV Benchmark Results\n");
    printf("Matrix: %d × %d, NNZ: %d\n", coo.m, coo.n, coo.nnz);
    printf("Grid size: %d, Block size: %d\n", (coo.m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE);
    printf("Warmup cycles: %d, Iterations: %d\n", warmup, iterations);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Benchmark kernels
    std::vector<KernelStats> results;
    
    printf("Benchmarking Kernel 1: CSR-Vector...\n");
    results.push_back(timeKernel(
        csrSpMV_Vector, coo.m, coo.nnz,
        csr.d_row_ptr, csr.d_col_idx, csr.d_values, d_x, d_y,
        warmup, iterations, "CSR-Vector"
    ));

    printf("Benchmarking Kernel 2: CSR-Flat (Warp-level)...\n");
    results.push_back(timeKernel(
        csrSpMV_Flat, coo.m, coo.nnz,
        csr.d_row_ptr, csr.d_col_idx, csr.d_values, d_x, d_y,
        warmup, iterations, "CSR-Flat"
    ));

    printf("Benchmarking Kernel 3: CSR-Line-Enhance...\n");
    results.push_back(timeKernel(
        csrSpMV_LineEnhance, coo.m, coo.nnz,
        csr.d_row_ptr, csr.d_col_idx, csr.d_values, d_x, d_y,
        warmup, iterations, "CSR-LineEnhance"
    ));

    // Print results
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Benchmark Results Summary\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    for (const auto &stat : results) {
        printf("%-25s: %8.4f ms | %10.2f GFLOP/s\n", stat.name, stat.avg_time_ms, stat.gflops);
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

    printf("\n✓ Best performing kernel: %s (%.2f GFLOP/s)\n\n", results[best_idx].name, best_gflops);

    // ==================== CPU VALIDATION ====================
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Validating against CPU baseline...\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Copy result from GPU for first kernel
    std::vector<FloatType> gpu_result(coo.m);
    csrSpMV_Vector<<<(coo.m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        coo.m, csr.d_row_ptr, csr.d_col_idx, csr.d_values, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_y, coo.m * sizeof(FloatType), cudaMemcpyDeviceToHost));

    // Compute CPU baseline
    std::vector<FloatType> h_x(coo.n);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, coo.n * sizeof(FloatType), cudaMemcpyDeviceToHost));
    
    std::vector<FloatType> cpu_result(coo.m);
    csrSpMV_CPU(coo.m, coo.n, csr.row_ptr.data(), csr.col_idx.data(), 
                csr.values.data(), h_x.data(), cpu_result.data());

    // Validate
    validateResult(coo.m, gpu_result.data(), cpu_result.data(), 1e-4);

    // Cleanup
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    freeCSRMatrix(csr);

    printf("\n✓ CSR benchmark completed successfully!\n\n");
    return 0;
}
