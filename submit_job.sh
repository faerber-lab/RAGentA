#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --job-name=main-rag

module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5
module load PyTorch/2.1.2

source env/bin/activate

# Run benchmark
python main.py --model "mistralai/Mistral-7B-v0.1" --benchmark all --limit 100
