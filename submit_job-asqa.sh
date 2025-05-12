#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=main-rag-asqa

# Set cache directories BEFORE activating environment
export HF_DATASETS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_datasets"
export TRANSFORMERS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_models"
export HF_HOME="/data/horse/ws/jihe529c-main-rag/cache/huggingface"
export TORCH_HOME="/data/horse/ws/jihe529c-main-rag/cache/torch"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HUGGING_FACE_HUB_TOKEN=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

# Create directories
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5
module load PyTorch/2.1.2

source env/bin/activate

# Run benchmark
python main.py --model "mistralai/Mistral-7B-v0.1" --benchmark asqa --limit 100
