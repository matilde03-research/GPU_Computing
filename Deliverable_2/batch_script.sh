#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --job-name=multiGPUSpMV
#SBATCH --output=output/multiGPU-%j.out
#SBATCH --error=output/multiGPU-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a30.24:2
#SBATCH --mem=8G

# Load modules
module load OpenMPI
module load CUDA/12.5.0

# Print compiler versions
gcc --version
mpicc --version

# mpirun -np 4 ./bin/spmv ../data/boyd2/boyd2.mtx 4 100 4
mpirun -np 2 ./bin/spmv ../data/ASIC_680ks.mtx 4 100 2
