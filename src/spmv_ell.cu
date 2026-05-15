#include "../include/mtx_parser.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 256
#define SHARED_MEM_SIZE 1024

// ==================== ELL KERNELS ====================

// Kernel 1: ELL-Basic (straightforward parallelization - one thread per row)
__global__ void ellSpMV_Basic(int m, int max_row_len, const int *col_idx, 
                               const FloatType *values, const FloatType *x, FloatType *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        FloatType sum = 0.0f;
        for (int j = 0; j < max_row_len; j++) {
            int col_id = col_idx[j * m + row];  // Column-major access
            if (col_id >= 0) {  // -1 indicates padding
                sum += values[j * m + row] * x[col_id];
            }
        }
        y[row] = sum;
    }
}

// Kernel 2: ELL-Optimized (shared memory + improved load balancing)
// Uses shared memory to cache column indices and better memory access patterns
__global__ void ellSpMV_Optimized(int m, int max_row_len, const int *col_idx, 
                                   const FloatType *values, const FloatType *x, FloatType *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    
    // Shared memory for caching data
    __shared__ FloatType s_values[SHARED_MEM_SIZE / sizeof(FloatType)];
    __shared__ int s_col_idx[SHARED_MEM_SIZE / sizeof(int)];
    
    if (row < m) {
        FloatType sum = 0.0f;
        
        // Process in chunks to utilize shared memory
        for (int chunk = 0; chunk < max_row_len; chunk += blockDim.x) {
            if (chunk + tx < max_row_len) {
                int idx = (chunk + tx) * m + row;
                s_col_idx[tx] = col_idx[idx];
                s_values[tx] = values[idx];
            }
            __syncthreads();
            
            // Compute with cached data
            if (chunk + tx < max_row_len) {
                int col_id = s_col_idx[tx];
                if (col_id >= 0) {
                    sum += s_values[tx] * x[col_id];
                }
            }
            __syncthreads();
        }
        
        y[row] = sum;
    }
}

// ==================== TIMING UTILITIES ====================

struct KernelStats {
    const char *name;
    double avg_time_ms;
    double gflops;
};

KernelStats timeKernel(void (*kernel)(int, int, const int*, const FloatType*, const FloatType*, FloatType*),
                       int m, int nnz, int max_row_len,
                       const int *d_col_idx, const FloatType *d_values,
                       const FloatType *d_x, FloatType *d_y,
                       int warmup, int iterations, const char *name) {
    // Warm-up
    for (int i = 0; i < warmup; i++) {
        kernel<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(m, max_row_len, d_col_idx, d_values, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(m, max_row_len, d_col_idx, d_values, d_x, d_y);
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
    printf("║           ELL Format SpMV Benchmark                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    // Read matrix
    printf("Loading matrix from: %s\n", mtx_file);
    COOMatrix coo = readMatrixMarket(mtx_file);
    printMatrixStats(coo);

    // Convert to ELL (with timing)
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Converting to ELL format...\n");
    ELLMatrix ell = cooToELL(coo);
    printf("Max row length (after padding): %d\n", ell.max_row_len);
    
    double memory_ell = (long long)coo.m * ell.max_row_len * (sizeof(int) + sizeof(FloatType)) / 1024.0;
    printf("Memory (ELL format): %.2f KB\n", memory_ell);

    // Allocate vectors on device
    FloatType *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, coo.n * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_y, coo.m * sizeof(FloatType)));

    // Generate random vector
    printf("Generating random vector (seed=42)...\n");
    generateRandomVector(d_x, coo.n, 42);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("ELL SpMV Benchmark Results\n");
    printf("Matrix: %d × %d, NNZ: %d\n", coo.m, coo.n, coo.nnz);
    printf("ELL Max row length: %d\n", ell.max_row_len);
    printf("Grid size: %d, Block size: %d\n", (coo.m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE);
    printf("Warmup cycles: %d, Iterations: %d\n", warmup, iterations);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Benchmark kernels
    std::vector<KernelStats> results;
    
    printf("Benchmarking Kernel 1: ELL-Basic...\n");
    results.push_back(timeKernel(
        ellSpMV_Basic, coo.m, coo.nnz, ell.max_row_len,
        ell.d_col_idx, ell.d_values, d_x, d_y,
        warmup, iterations, "ELL-Basic"
    ));

    printf("Benchmarking Kernel 2: ELL-Optimized (shared memory)...\n");
    results.push_back(timeKernel(
        ellSpMV_Optimized, coo.m, coo.nnz, ell.max_row_len,
        ell.d_col_idx, ell.d_values, d_x, d_y,
        warmup, iterations, "ELL-Optimized"
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

    // For ELL validation, we need to convert back to CSR for easy CPU comparison
    // Or compute expected result directly from COO
    std::vector<FloatType> h_x(coo.n);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, coo.n * sizeof(FloatType), cudaMemcpyDeviceToHost));

    // Compute reference from COO in CSR format
    CSRMatrix csr_ref = cooToCSR(coo);
    std::vector<FloatType> cpu_result(coo.m);
    csrSpMV_CPU(coo.m, coo.n, csr_ref.row_ptr.data(), csr_ref.col_idx.data(), 
                csr_ref.values.data(), h_x.data(), cpu_result.data());

    // Copy GPU result
    std::vector<FloatType> gpu_result(coo.m);
    ellSpMV_Basic<<<(coo.m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        coo.m, ell.max_row_len, ell.d_col_idx, ell.d_values, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_y, coo.m * sizeof(FloatType), cudaMemcpyDeviceToHost));

    // Validate
    validateResult(coo.m, gpu_result.data(), cpu_result.data(), 1e-4);
    
    freeCSRMatrix(csr_ref);

    // Cleanup
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    freeELLMatrix(ell);

    printf("\n✓ ELL benchmark completed successfully!\n\n");
    return 0;
}
