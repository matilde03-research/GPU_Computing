# Deliverable_2

```
Deliverable_2/
├── bin/ --> will be created and populated after *make BINUTILS=/bin*
├── CompleteResults/ --> files are not listed here because they're a lot but it's a folder with detailed output
│                        for the 10 Suite Sparse matrices (the outputs for each matrix are 3, one per GPUs number
│                        (1,2,4) and then there are also the results for 4 randomly generated matrices (2 from R-MAT
│                        and 2 from stencil)
├── include/
│    ├── mtx_parser.h
├── src/
│    ├── mtx_parser.cu
│    ├── spmv.cu      
├── Makefile
├── README.md
└── batch_script.sh //To run the programs in a job in the assigned environment
```

## Building

### Prerequisites

- **CUDA Toolkit** 12.5 or later
- **OpenMPI** 
- **NVIDIA GPU** with compute capability ≥ 7.5 (tested on A30, RTX-class GPUs)
- **Make** build tool
- **g++** or compatible C++ compiler

### Compilation

```bash
# Just the first time
module load OpenMPI && module load CUDA/12.5.0

# Build all executables
make BINUTILS=/bin

# Run experiments
sbatch batch_script.sh
```
The batch_script.sh file is the one used during class labs and it will generate an output and an error file in a directory called "output" for each job launched

Important! Before executing the sbatch command is important to check for the some lines of the file that is what actually launches the job in the cluster
The lines to check/change are:
```bash
#SBATCH --ntasks=2 --> change the number according to how many gpus you want to use for the experiment

#SBATCH --gres=gpu:a30.24:2 --> change the last number according to how many gpus you want to use for the experiment
#SBATCH --mem=8G --> safe for our matrices

mpirun -np 2 ./bin/spmv ../data/ASIC_680ks.mtx 4 100 2   --> explained later
```

the first number is the number of processes, then there's the executable program, then the warmup cycles, the number of iterations and again the number of gpus you want to use
You can also replace the Suite Sparse matrix path with a randomly generated matrix, instead of the path you can insert *random_rmat* or *random_stencil* according to the type of matrix you want to experiment with

After that you run the batch_script.sh you'll have 2 output file to inspect in the "output" directory that will be created

The Suite Sparse matrices are the one already downloaded for the first assignment, to download them you can use the download.sh file present one level outside this folder. It will download the matrices in a directory called "data". The provided line of code for the run in the batch_script.sh should be able to detect the data folder even if it's "one folder above".

