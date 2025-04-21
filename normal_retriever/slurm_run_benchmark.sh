#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --job-name=main-rag-gen

# Set cache directories BEFORE activating environment
export HF_DATASETS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_datasets"
export TRANSFORMERS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_models"
export HF_HOME="/data/horse/ws/jihe529c-main-rag/cache/huggingface"
export TORCH_HOME="/data/horse/ws/jihe529c-main-rag/cache/torch"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HUGGING_FACE_HUB_TOKEN=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
export CUDA_LAUNCH_BLOCKING=1

# Create directories
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5
module load PyTorch/2.1.2

source env/bin/activate

# Run the benchmark with specified parameters
BENCHMARK="triviaqa"  # Options: arc, triviaqa, popqa, asqa, all
MODEL="mistralai/Mistral-7B-v0.1"
N_VALUE=0.0
LIMIT=100  # Set to None for full dataset

# Run with srun to ensure proper GPU allocation
srun python benchmark_evaluation.py \
    --model $MODEL \
    --benchmark $BENCHMARK \
    --n $N_VALUE \
    --limit $LIMIT
    > logs/${BENCHMARK}_${N_VALUE}_${LIMIT}.log 2>&1

echo "Job completed"
