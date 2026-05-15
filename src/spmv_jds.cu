#include "../include/mtx_parser.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ==================== JDS KERNELS ====================

// Kernel 1: JDS-Basic (straightforward row-sorted parallelization)
__global__ void jdsSpMV_Basic(int m, const int *perm, const int *col_start, 
                               const int *diag_len, const int *col_idx, 
                               const FloatType *values, const FloatType *x, FloatType *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        int orig_row = perm[idx];
        FloatType sum = 0.0f;
        
        // Iterate through all diagonals for this row
        for (int d = 0; d < m; d++) {
            if (idx < diag_len[d]) {
                int pos = col_start[d] + idx;
                sum += values[pos] * x[col_idx[pos]];
            }
        }
        
        y[orig_row] = sum;
    }
}

// Kernel 2: JDS-Optimized (warp-level parallelism + shared memory)
__global__ void jdsSpMV_Optimized(int m, const int *perm, const int *col_start, 
                                   const int *diag_len, const int *col_idx, 
                                   const FloatType *values, const FloatType *x, FloatType *y) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane = thread_id % WARP_SIZE;
    
    // Shared memory for caching
    __shared__ FloatType s_values[256];
    __shared__ int s_col_idx[256];
    
    if (warp_id < m) {
        int idx = warp_id;
        int orig_row = perm[idx];
        FloatType sum = 0.0f;
        
        // Process diagonals (each warp processes one row)
        for (int d = 0; d < m; d++) {
            if (idx < diag_len[d]) {
                int diag_start = col_start[d];
                int pos = diag_start + idx;
                
                // Cache value and column index in shared memory
                if (lane == 0) {
                    s_values[threadIdx.x] = values[pos];
                    s_col_idx[threadIdx.x] = col_idx[pos];
                }
                __syncthreads();
                
                // Use cached data for computation
                sum += s_values[threadIdx.x] * x[s_col_idx[threadIdx.x]];
            }
        }
        
        if (lane == 0) {
            y[orig_row] = sum;
        }
    }
}

// ==================== TIMING UTILITIES ====================

struct KernelStats {
    const char *name;
    double avg_time_ms;
    double gflops;
};

KernelStats timeKernel(void (*kernel)(int, const int*, const int*, const int*, const int*, 
                                       const FloatType*, const FloatType*, FloatType*),
                       int m, int nnz,
                       const int *d_perm, const int *d_col_start, 
                       const int *d_diag_len, const int *d_col_idx, const FloatType *d_values,
                       const FloatType *d_x, FloatType *d_y,
                       int warmup, int iterations, const char *name) {
    // Warm-up
    for (int i = 0; i < warmup; i++) {
        kernel<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(m, d_perm, d_col_start, 
                                                                    d_diag_len, d_col_idx, 
                                                                    d_values, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(m, d_perm, d_col_start, 
                                                                    d_diag_len, d_col_idx, 
                                                                    d_values, d_x, d_y);
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
    printf("║           JDS Format SpMV Benchmark                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    // Read matrix
    printf("Loading matrix from: %s\n", mtx_file);
    COOMatrix coo = readMatrixMarket(mtx_file);
    printMatrixStats(coo);

    // Convert to JDS (with timing)
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Converting to JDS format...\n");
    JDSMatrix jds = cooToJDS(coo);

    // Allocate vectors on device
    FloatType *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, coo.n * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_y, coo.m * sizeof(FloatType)));

    // Generate random vector
    printf("Generating random vector (seed=42)...\n");
    generateRandomVector(d_x, coo.n, 42);

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("JDS SpMV Benchmark Results\n");
    printf("Matrix: %d × %d, NNZ: %d\n", coo.m, coo.n, coo.nnz);
    printf("Grid size: %d, Block size: %d\n", (coo.m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE);
    printf("Warmup cycles: %d, Iterations: %d\n", warmup, iterations);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // Benchmark kernels
    std::vector<KernelStats> results;
    
    printf("Benchmarking Kernel 1: JDS-Basic...\n");
    results.push_back(timeKernel(
        jdsSpMV_Basic, coo.m, coo.nnz,
        jds.d_perm, jds.d_col_start, jds.d_diag_len, jds.d_col_idx, jds.d_values,
        d_x, d_y,
        warmup, iterations, "JDS-Basic"
    ));

    printf("Benchmarking Kernel 2: JDS-Optimized (warp + shared mem)...\n");
    results.push_back(timeKernel(
        jdsSpMV_Optimized, coo.m, coo.nnz,
        jds.d_perm, jds.d_col_start, jds.d_diag_len, jds.d_col_idx, jds.d_values,
        d_x, d_y,
        warmup, iterations, "JDS-Optimized"
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

    // For JDS validation, we use CSR reference from COO
    std::vector<FloatType> h_x(coo.n);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, coo.n * sizeof(FloatType), cudaMemcpyDeviceToHost));

    // Compute reference from CSR
    CSRMatrix csr_ref = cooToCSR(coo);
    std::vector<FloatType> cpu_result(coo.m);
    csrSpMV_CPU(coo.m, coo.n, csr_ref.row_ptr.data(), csr_ref.col_idx.data(), 
                csr_ref.values.data(), h_x.data(), cpu_result.data());

    // Copy GPU result
    std::vector<FloatType> gpu_result(coo.m);
    jdsSpMV_Basic<<<(coo.m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        coo.m, jds.d_perm, jds.d_col_start, jds.d_diag_len, jds.d_col_idx, jds.d_values, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_y, coo.m * sizeof(FloatType), cudaMemcpyDeviceToHost));

    // Validate
    validateResult(coo.m, gpu_result.data(), cpu_result.data(), 1e-4);
    
    freeCSRMatrix(csr_ref);

    // Cleanup
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    freeJDSMatrix(jds);

    printf("\n✓ JDS benchmark completed successfully!\n\n");
    return 0;
}
