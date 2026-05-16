#include "../include/mtx_parser.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cusparse.h>

#define BLOCK_SIZE 256

// ==================== CUSPARSE HELPER ====================

struct cusparseHandle_guard {
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    
    cusparseHandle_guard() : handle(nullptr), matA(nullptr), vecX(nullptr), vecY(nullptr) {}
    
    ~cusparseHandle_guard() {
        if (vecY) cusparseDestroyDnVec(vecY);
        if (vecX) cusparseDestroyDnVec(vecX);
        if (matA) cusparseDestroySpMat(matA);
        if (handle) cusparseDestroy(handle);
    }
};

// ==================== VALIDATION ====================

bool validateResult(const std::vector<FloatType>& gpu_result, 
                   const std::vector<FloatType>& cpu_result,
                   int m, const char *kernel_name) {
    FloatType tolerance = 1e-3;  // absolute floor
    FloatType rel_tolerance = 1e-4;  // relative tolerance
    int error_count = 0;
    FloatType max_error = 0.0f;

    for (int i = 0; i < m; i++) {
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
};

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
    printf("║        cuSPARSE CSR Format SpMV Benchmark                 ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    // Read matrix
    printf("Loading matrix from: %s\n", mtx_file);
    COOMatrix coo = readMatrixMarket(mtx_file);
    
    

    // Convert to CSR
    printf("\nConverting to CSR format...\n");
    CSRMatrix csr = cooToCSR(coo);

    printf("CSR Device Pointers - RowPtr: %p, ColIdx: %p, Values: %p\n", 
       (void*)csr.d_row_ptr, (void*)csr.d_col_idx, (void*)csr.d_values);

    // Allocate vectors on device
    FloatType *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, coo.n * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_y, coo.m * sizeof(FloatType)));

    // Generate random vector (same seed as other benchmarks)
    printf("Generating random vector (seed=42)...\n");
    generateRandomVector(d_x, coo.n, 42);

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("cuSPARSE SpMV Setup\n");
    printf("Matrix: %d × %d, NNZ: %d\n", coo.m, coo.n, coo.nnz);
    printf("Warmup cycles: %d, Iterations: %d\n", warmup, iterations);
    printf("════════════════════════════════════════════════════════════\n\n");

    // Setup cuSPARSE
    auto setup_start = std::chrono::high_resolution_clock::now();
    
    cusparseHandle_guard sparse;
    cusparseCreate(&sparse.handle);
    
    // Create sparse matrix descriptor
    cusparseCreateCsr(&sparse.matA, coo.m, coo.n, coo.nnz,
                      csr.d_row_ptr, csr.d_col_idx, csr.d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    printf("d_y pointer: %p\n", (void*)d_y);
    
    // Create dense vectors
    cusparseCreateDnVec(&sparse.vecX, coo.n, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&sparse.vecY, coo.m, d_y, CUDA_R_32F);
    
    // Allocate workspace
    size_t bufferSize = 0;
    FloatType one = 1.0f, zero = 0.0f;
    cusparseSpMV_bufferSize(sparse.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &one, sparse.matA, sparse.vecX, &zero, sparse.vecY,
                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    
    void *dBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
    
    auto setup_end = std::chrono::high_resolution_clock::now();
    double setup_time = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
    printf("cuSPARSE setup time: %.4f ms\n", setup_time);
    
    CUDA_CHECK(cudaMemset(d_y, 0, coo.m * sizeof(FloatType)));

    // Warm-up
    printf("\nWarm-up phase...\n");
    for (int i = 0; i < warmup; i++) {
        cusparseSpMV(sparse.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, sparse.matA, sparse.vecX, &zero, sparse.vecY,
                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    printf("Benchmarking cuSPARSE SpMV...\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        cusparseSpMV(sparse.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, sparse.matA, sparse.vecX, &zero, sparse.vecY,
                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    
    double avg_time = time_ms / iterations;
    double gflops = (2.0 * coo.nnz) / (avg_time * 1e-3) / 1e9;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Print results
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("cuSPARSE SpMV Results\n");
    printf("════════════════════════════════════════════════════════════\n\n");
    printf("%-25s: %8.4f ms | %10.2f GFLOP/s\n", "cuSPARSE SpMV", avg_time, gflops);

    // ==================== VALIDATION ====================
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("Validation Against CPU Baseline (OpenMP)\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    // Allocate CPU vectors
    std::vector<FloatType> h_x(coo.n);
    std::vector<FloatType> h_y_gpu(coo.m);
    std::vector<FloatType> h_y_cpu(coo.m, 0.0f);

    // Copy the SAME random vector used on GPU (generated with seed=42)
    printf("Copying GPU input vector x to host...\n");
    

    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, coo.n * sizeof(FloatType), cudaMemcpyDeviceToHost));

    // Run CPU baseline with OpenMP using the same random vector
    printf("Running CPU baseline with OpenMP...\n");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < coo.nnz; i++) {
        int row = coo.row_indices[i];
        int col = coo.col_indices[i];
        #pragma omp atomic
        h_y_cpu[row] += coo.values[i] * h_x[col];
    }

    // Copy GPU result
    printf("Copying GPU output vector y to host...\n");
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, coo.m * sizeof(FloatType), cudaMemcpyDeviceToHost));

    // Validate results
    printf("Comparing GPU and CPU results...\n");
    printf("───────────────────────────────────────────────────────────\n");
    bool valid = validateResult(h_y_gpu, h_y_cpu, coo.m, "cuSPARSE");

    if (valid) {
        printf("\n✓ Validation PASSED: GPU results match CPU baseline\n\n");
    } else {
        printf("\n✗ Validation FAILED: GPU results differ from CPU baseline\n\n");
    }

    printf("════════════════════════════════════════════════════════════\n");

    // Cleanup
    CUDA_CHECK(cudaFree(dBuffer));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    freeCSRMatrix(csr);

    printf("✓ cuSPARSE benchmark completed successfully!\n\n");
    return 0;
}
