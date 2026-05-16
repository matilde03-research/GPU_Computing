#include "../include/mtx_parser.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#define BLOCK_SIZE 256
#define X_TILE_SIZE BLOCK_SIZE

//============================================================
// VALIDATION
//============================================================

bool validateResult(const std::vector<FloatType>& gpu_result, 
                   const std::vector<FloatType>& cpu_result,
                   int m, const char *kernel_name) {
    FloatType tolerance = 1e-3;
    FloatType rel_tolerance = 1e-4;
    int error_count = 0;
    FloatType max_error = 0.0f;

    for (int i = 0; i < m; i++) {
        FloatType diff = fabs(gpu_result[i] - cpu_result[i]);
        FloatType rel_error = diff / (fabs(cpu_result[i]) + 1e-10f);
        if (diff > tolerance && rel_error > rel_tolerance) {
            error_count++;
            max_error = fmax(max_error, diff);
            if (error_count <= 5) {
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

//============================================================
// KERNEL 1 : Basic ELL (Optimized)
// One thread computes one row
// Optimizations:
//  - __ldg() for cached x vector reads
//  - Early exit for invalid entries
//  - Coalesced memory access patterns
//============================================================

__global__
void ellSpMV_Basic(
    int m,
    int max_row_len,
    const int* __restrict__ col_idx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= m)
        return;

    FloatType sum = 0.0f;

    // Process all entries in this row (stored in ELL format)
    for(int j = 0; j < max_row_len; j++)
    {
        // In ELL format, entry (j, row) is stored at index j*m + row
        // This is row-major storage by column position
        int idx = j * m + row;

        int col = col_idx[idx];

        // Skip invalid entries (ELL pads with -1)
        if(col != -1)
        {
            FloatType val = values[idx];
            // Use __ldg() for L1 cache-optimized reads of x vector
            sum += val * __ldg(&x[col]);
        }
    }

    y[row] = sum;
}


//============================================================
// KERNEL 2 : Coalesced ELL
//
// Improved variant: Multiple threads cooperate on one row
// using warp-level shuffle for efficient reduction.
//
// Strategy:
//  - Warp (32 threads) processes one row
//  - Each thread reads max_row_len/32 entries from that row
//  - Warp shuffle reduction to combine partial sums
//  - Only lane 0 writes final result
//============================================================

__global__
void ellSpMV_Coalesced(
    int m,
    int max_row_len,
    const int* __restrict__ col_idx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* y)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if(warp_id >= m)
        return;

    int row = warp_id;
    FloatType sum = 0.0f;

    // Each lane processes entries in this row, strided by warp size
    for(int j = lane; j < max_row_len; j += 32)
    {
        int idx = j * m + row;
        int col = col_idx[idx];

        if(col != -1)
        {
            FloatType val = values[idx];
            sum += val * __ldg(&x[col]);
        }
    }

    // Warp-level reduction via shuffle down
    for(int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    // Lane 0 writes the result
    if(lane == 0)
        y[row] = sum;
}


//============================================================
// BENCHMARK STRUCT
//============================================================

struct KernelStats
{
    const char* name;
    double conversion_time_ms;   // COO->ELL conversion time
    double computation_time_ms;  // Just the kernel execution
    double total_time_ms;        // Total time (conversion + computation)
    double gflops;
};


//============================================================
// BENCHMARK TEMPLATE
//============================================================

template<typename Kernel>
KernelStats timeKernel(
        Kernel kernel,
        int m,
        int nnz,
        int max_row_len,
        const int *d_col_idx,
        const FloatType *d_values,
        const FloatType *d_x,
        FloatType *d_y,
        double conversion_time_ms,
        int warmup,
        int iterations,
        const char *name)
{
    int grid = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

    ////////////////////////////////////////////
    // Warmup
    ////////////////////////////////////////////

    for(int i = 0; i < warmup; i++)
    {
        CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));
        kernel<<<grid, BLOCK_SIZE>>>(
            m,
            max_row_len,
            d_col_idx,
            d_values,
            d_x,
            d_y
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    ////////////////////////////////////////////
    // Timing
    ////////////////////////////////////////////

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Clear output before timing
    CUDA_CHECK(cudaMemset(d_y, 0, m * sizeof(FloatType)));

    CUDA_CHECK(cudaEventRecord(start));

    for(int i = 0; i < iterations; i++)
    {
        kernel<<<grid, BLOCK_SIZE>>>(
            m,
            max_row_len,
            d_col_idx,
            d_values,
            d_x,
            d_y
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    ms /= iterations;

    double gflops = (2.0 * nnz) / (ms * 1e6);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    KernelStats result;
    result.name = name;
    result.conversion_time_ms = conversion_time_ms;
    result.computation_time_ms = ms;
    result.total_time_ms = conversion_time_ms + ms;
    result.gflops = gflops;

    return result;
}



//============================================================
// MAIN
//============================================================

int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        fprintf(stderr, "Usage: %s <matrix.mtx> <warmup> <iterations>\n", argv[0]);
        return 1;
    }

    const char* mtx_file = argv[1];
    int warmup = atoi(argv[2]);
    int iterations = atoi(argv[3]);

    if(warmup <= 0 || iterations <= 0)
    {
        fprintf(stderr, "Error: arguments must be positive\n");
        return 1;
    }

    printf("\n");
    printf("════════════════════════════════════════════════════════\n");
    printf("        ELL SpMV Benchmark (Basic + Coalesced)\n");
    printf("════════════════════════════════════════════════════════\n");

    ////////////////////////////////////////////
    // Load matrix
    ////////////////////////////////////////////

    printf("\nLoading: %s\n", mtx_file);
    COOMatrix coo = readMatrixMarket(mtx_file);

    ////////////////////////////////////////////
    // Convert and time
    ////////////////////////////////////////////

    printf("\nConverting COO -> ELL...\n");
    
    cudaEvent_t conv_start, conv_stop;
    CUDA_CHECK(cudaEventCreate(&conv_start));
    CUDA_CHECK(cudaEventCreate(&conv_stop));
    CUDA_CHECK(cudaEventRecord(conv_start));

    ELLMatrix ell = cooToELL(coo);

    CUDA_CHECK(cudaEventRecord(conv_stop));
    CUDA_CHECK(cudaEventSynchronize(conv_stop));
    
    float conversion_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&conversion_ms, conv_start, conv_stop));
    CUDA_CHECK(cudaEventDestroy(conv_start));
    CUDA_CHECK(cudaEventDestroy(conv_stop));

    printf("Max row length: %d\n", ell.max_row_len);
    printf("Format conversion time (COO->ELL): %.4f ms\n", conversion_ms);

    double memory_ell = (long long)coo.m * ell.max_row_len * 
                         (sizeof(int) + sizeof(FloatType)) / 1024.0 / 1024.0;
    printf("ELL Memory: %.2f MB\n", memory_ell);

    ////////////////////////////////////////////
    // Allocate GPU memory for ELL matrix
    ////////////////////////////////////////////

    printf("\nAllocating GPU memory for ELL matrix...\n");
    
    CUDA_CHECK(cudaMalloc(&ell.d_col_idx, 
                         (long long)coo.m * ell.max_row_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ell.d_values, 
                         (long long)coo.m * ell.max_row_len * sizeof(FloatType)));

    // Copy ELL matrix to GPU
    printf("Copying ELL matrix to GPU...\n");
    CUDA_CHECK(cudaMemcpy(ell.d_col_idx, ell.col_idx.data(),
                         (long long)coo.m * ell.max_row_len * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ell.d_values, ell.values.data(),
                         (long long)coo.m * ell.max_row_len * sizeof(FloatType),
                         cudaMemcpyHostToDevice));

    ////////////////////////////////////////////
    // Allocate vectors
    ////////////////////////////////////////////

    FloatType* d_x;
    FloatType* d_y;

    printf("Allocating vectors...\n");
    CUDA_CHECK(cudaMalloc(&d_x, coo.n * sizeof(FloatType)));
    CUDA_CHECK(cudaMalloc(&d_y, coo.m * sizeof(FloatType)));

    ////////////////////////////////////////////
    // Input vector
    ////////////////////////////////////////////

    printf("Generating random vector (seed=42)...\n");
    generateRandomVector(d_x, coo.n, 42);

    ////////////////////////////////////////////
    // Benchmark
    ////////////////////////////////////////////

    std::vector<KernelStats> results;

    printf("\n════════════════════════════════════════════════════════\n");
    printf("Matrix: %d × %d, NNZ: %d\n", coo.m, coo.n, coo.nnz);
    printf("Warmup cycles: %d, Iterations: %d\n", warmup, iterations);
    printf("════════════════════════════════════════════════════════\n");

    printf("\nBenchmarking ELL-Basic...\n");
    results.push_back(
        timeKernel(
            ellSpMV_Basic,
            coo.m,
            coo.nnz,
            ell.max_row_len,
            ell.d_col_idx,
            ell.d_values,
            d_x,
            d_y,
            conversion_ms,
            warmup,
            iterations,
            "ELL-Basic"
        )
    );

    printf("Benchmarking ELL-Coalesced...\n");
    results.push_back(
        timeKernel(
            ellSpMV_Coalesced,
            coo.m,
            coo.nnz,
            ell.max_row_len,
            ell.d_col_idx,
            ell.d_values,
            d_x,
            d_y,
            conversion_ms,
            warmup,
            iterations,
            "ELL-Coalesced"
        )
    );

    ////////////////////////////////////////////
    // Results
    ////////////////////////////////////////////

    printf("\n════════════════════════════════════════════════════════\n");
    printf("Performance Results – Detailed Breakdown\n");
    printf("════════════════════════════════════════════════════════\n\n");

    for(auto& r : results)
    {
        printf("%-20s:\n", r.name);
        printf("  Format conversion: %8.4f ms\n", r.conversion_time_ms);
        printf("  Computation:       %8.4f ms\n", r.computation_time_ms);
        printf("  Total:             %8.4f ms | %10.2f GFLOP/s\n\n",
               r.total_time_ms, r.gflops);
    }

    ////////////////////////////////////////////
    // Find best
    ////////////////////////////////////////////

    int best = 0;
    for(int i = 1; i < results.size(); i++)
    {
        if(results[i].gflops > results[best].gflops)
        {
            best = i;
        }
    }

    printf("Best kernel: %s (%.2f GFLOP/s)\n",
           results[best].name, results[best].gflops);

    double speedup = results[best].gflops / results[0].gflops;
    printf("Speedup vs Basic: %.2fx\n\n", speedup);

    ////////////////////////////////////////////
    // Validation
    ////////////////////////////////////////////

    printf("════════════════════════════════════════════════════════\n");
    printf("Validation Against CPU Baseline (COO-based with OpenMP)\n");
    printf("════════════════════════════════════════════════════════\n\n");

    // Allocate CPU vectors
    std::vector<FloatType> h_x(coo.n);
    std::vector<FloatType> h_y_gpu(coo.m);
    std::vector<FloatType> h_y_cpu(coo.m, 0.0f);

    // Copy input vector from GPU
    printf("Copying GPU input vector x to host...\n");
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, coo.n * sizeof(FloatType), 
                         cudaMemcpyDeviceToHost));

    // Run one more kernel on GPU for validation (ELL-Basic)
    printf("Running GPU kernel for validation...\n");
    int grid = (coo.m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMemset(d_y, 0, coo.m * sizeof(FloatType)));
    ellSpMV_Basic<<<grid, BLOCK_SIZE>>>(
        coo.m, ell.max_row_len, ell.d_col_idx, ell.d_values, d_x, d_y
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy GPU result
    printf("Copying GPU output vector y to host...\n");
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, coo.m * sizeof(FloatType), 
                         cudaMemcpyDeviceToHost));

    // Run CPU baseline using COO format with OpenMP (same as cusparse/coo)
    printf("Running CPU baseline with OpenMP (COO format)...\n");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < coo.nnz; i++) {
        int row = coo.row_indices[i];
        int col = coo.col_indices[i];
        #pragma omp atomic
        h_y_cpu[row] += coo.values[i] * h_x[col];
    }

    // Validate
    printf("Comparing GPU and CPU results...\n");
    printf("───────────────────────────────────────────────────────\n");
    bool valid = validateResult(h_y_gpu, h_y_cpu, coo.m, "ELL-Basic");

    if (valid) {
        printf("\n Validation PASSED: GPU results match CPU baseline\n\n");
    } else {
        printf("\n Validation FAILED: GPU results differ from CPU baseline\n\n");
    }

    ////////////////////////////////////////////
    // Cleanup
    ////////////////////////////////////////////

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    freeELLMatrix(ell);

    printf("════════════════════════════════════════════════════════\n");
    printf(" Benchmark completed successfully!\n\n");

    return 0;
}
