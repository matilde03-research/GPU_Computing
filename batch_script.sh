#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

#SBATCH --job-name=test
#SBATCH --output=output/test-%j.out
#SBATCH --error=output/test-%j.err

module load CUDA/11.8.0

./bin/spmv_cusparse ./ASIC_680ks.mtx 4 100