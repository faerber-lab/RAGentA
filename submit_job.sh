#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=main-rag

# Set cache directories on horse workspace
CACHE_BASE="/data/horse/ws/${USER}-main-rag/cache"
mkdir -p $CACHE_BASE

# Set all cache directories
export HF_DATASETS_CACHE="${CACHE_BASE}/hf_datasets"
export TRANSFORMERS_CACHE="${CACHE_BASE}/hf_models"
export HF_HOME="${CACHE_BASE}/huggingface"
export TORCH_HOME="${CACHE_BASE}/torch"
# Accept running custom code for datasets
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Create cache directories
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5
module load PyTorch/2.1.2

source env/bin/activate

# Run benchmark
python main.py --model "mistralai/Mistral-7B-v0.1" --benchmark all --limit 100
