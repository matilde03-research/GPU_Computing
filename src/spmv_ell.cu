#include "../include/mtx_parser.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define BLOCK_SIZE 256
#define X_TILE_SIZE BLOCK_SIZE

//============================================================
// KERNEL 1 : Basic ELL
// One thread computes one row
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
    int row =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(row >= m)
        return;

    FloatType sum = 0;

    for(int j=0;j<max_row_len;j++)
    {
        int idx = j*m + row;

        int col = col_idx[idx];

        if(col != -1)
        {
            sum +=
                values[idx] *
                __ldg(&x[col]);
        }
    }

    y[row] = sum;
}


//============================================================
// KERNEL 2 : Shared-memory version
//
// Cache x vector tiles in shared memory
//============================================================

__global__
void ellSpMV_Shmem(
    int m,
    int max_row_len,
    const int* __restrict__ col_idx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* y)
{
    int row =
        blockIdx.x*blockDim.x +
        threadIdx.x;

    int tx=threadIdx.x;

    if(row>=m)
        return;

    FloatType sum=0;

    __shared__
    FloatType sx[X_TILE_SIZE];

    //////////////////////////////////////////////////////
    // Tile through x vector
    //////////////////////////////////////////////////////

    for(int tile=0;
        tile<m;
        tile+=X_TILE_SIZE)
    {
        if(tile+tx<m)
        {
            sx[tx] =
                __ldg(&x[tile+tx]);
        }

        __syncthreads();

        for(int j=0;
            j<max_row_len;
            j++)
        {
            int idx =
                j*m+row;

            int col=
                col_idx[idx];

            if(col==-1)
                continue;

            FloatType val=
                values[idx];

            if(col>=tile &&
               col<tile+X_TILE_SIZE)
            {
                sum +=
                    val*
                    sx[col-tile];
            }
            else
            {
                sum +=
                    val*
                    __ldg(&x[col]);
            }
        }

        __syncthreads();
    }

    y[row]=sum;
}


//============================================================
// BENCHMARK STRUCT
//============================================================

struct KernelStats
{
    const char* name;

    double avg_time_ms;

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
        int warmup,
        int iterations,
        const char *name)
{
    int grid =
        (m+BLOCK_SIZE-1)
        /BLOCK_SIZE;

    ////////////////////////////////////////////
    // Warmup
    ////////////////////////////////////////////

    for(int i=0;i<warmup;i++)
    {
        kernel<<<grid,BLOCK_SIZE>>>(
            m,
            max_row_len,
            d_col_idx,
            d_values,
            d_x,
            d_y
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    ////////////////////////////////////////////
    // Timing
    ////////////////////////////////////////////

    cudaEvent_t start,stop;

    CUDA_CHECK(
        cudaEventCreate(&start)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop)
    );

    CUDA_CHECK(
        cudaEventRecord(start)
    );

    for(int i=0;i<iterations;i++)
    {
        kernel<<<grid,BLOCK_SIZE>>>(
            m,
            max_row_len,
            d_col_idx,
            d_values,
            d_x,
            d_y
        );
    }

    CUDA_CHECK(
        cudaEventRecord(stop)
    );

    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );

    float ms=0;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &ms,
            start,
            stop
        )
    );

    ms/=iterations;

    double gflops =
        (2.0*nnz)
        /(ms*1e6);

    CUDA_CHECK(
        cudaEventDestroy(start)
    );

    CUDA_CHECK(
        cudaEventDestroy(stop)
    );

    KernelStats result;

    result.name=name;
    result.avg_time_ms=ms;
    result.gflops=gflops;

    return result;
}



//============================================================
// MAIN
//============================================================

int main(int argc,char* argv[])
{
    if(argc!=4)
    {
        fprintf(
            stderr,
            "Usage: %s <matrix.mtx> <warmup> <iterations>\n",
            argv[0]);

        return 1;
    }

    const char* mtx_file=
        argv[1];

    int warmup=
        atoi(argv[2]);

    int iterations=
        atoi(argv[3]);

    if(warmup<=0 ||
       iterations<=0)
    {
        fprintf(
            stderr,
            "Error: arguments must be positive\n");

        return 1;
    }

    printf("\n");
    printf("====================================================\n");
    printf("            ELL SpMV Benchmark\n");
    printf("====================================================\n");


    ////////////////////////////////////////////
    // Load matrix
    ////////////////////////////////////////////

    printf(
        "\nLoading: %s\n",
        mtx_file);

    COOMatrix coo =
        readMatrixMarket(
            mtx_file);

    ////////////////////////////////////////////
    // Convert
    ////////////////////////////////////////////

    printf(
        "\nConverting COO -> ELL...\n");

    ELLMatrix ell =
        cooToELL(coo);

    printf(
        "Max row length: %d\n",
        ell.max_row_len);

    double memory_ell=
        (long long)
        coo.m*
        ell.max_row_len*
        (sizeof(int)+sizeof(FloatType))
        /1024.0;

    printf(
        "ELL Memory: %.2f KB\n",
        memory_ell);


    ////////////////////////////////////////////
    // Allocate vectors
    ////////////////////////////////////////////

    FloatType* d_x;
    FloatType* d_y;

    CUDA_CHECK(
        cudaMalloc(
            &d_x,
            coo.n*sizeof(FloatType)));

    CUDA_CHECK(
        cudaMalloc(
            &d_y,
            coo.m*sizeof(FloatType)));

    ////////////////////////////////////////////
    // Input vector
    ////////////////////////////////////////////

    printf(
        "\nGenerating vector...\n");

    generateRandomVector(
        d_x,
        coo.n,
        42);

    ////////////////////////////////////////////
    // Benchmark
    ////////////////////////////////////////////

    std::vector<KernelStats>
        results;

    printf(
        "\nBenchmarking Basic...\n");

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
            warmup,
            iterations,
            "ELL-Basic"
        )
    );



    printf(
        "Benchmarking Shared...\n");

    results.push_back(
        timeKernel(
            ellSpMV_Shmem,
            coo.m,
            coo.nnz,
            ell.max_row_len,
            ell.d_col_idx,
            ell.d_values,
            d_x,
            d_y,
            warmup,
            iterations,
            "ELL-Shared"
        )
    );


    ////////////////////////////////////////////
    // Results
    ////////////////////////////////////////////

    printf("\n");
    printf("====================================================\n");
    printf("Results\n");
    printf("====================================================\n\n");

    for(auto& r:results)
    {
        printf(
            "%-20s : %8.4f ms | %10.2f GFLOP/s\n",
            r.name,
            r.avg_time_ms,
            r.gflops);
    }


    ////////////////////////////////////////////
    // Find best
    ////////////////////////////////////////////

    int best=0;

    for(int i=1;
        i<results.size();
        i++)
    {
        if(results[i].gflops>
           results[best].gflops)
        {
            best=i;
        }
    }

    printf(
        "\nBest kernel: %s (%.2f GFLOP/s)\n",
        results[best].name,
        results[best].gflops
    );


    ////////////////////////////////////////////
    // Cleanup
    ////////////////////////////////////////////

    CUDA_CHECK(
        cudaFree(d_x));

    CUDA_CHECK(
        cudaFree(d_y));

    freeELLMatrix(ell);

    printf(
        "\nBenchmark completed.\n\n");

    return 0;
}