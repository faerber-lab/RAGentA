#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=main-rag-split
#SBATCH --output=logs/main_rag_split_%j.log

# Parameters passed from command line
SPLIT_FILE=$1
OUTPUT_PREFIX=$2

# Set cache directories
export HF_DATASETS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_datasets"
export TRANSFORMERS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_models"
export HF_HOME="/data/horse/ws/jihe529c-main-rag/cache/huggingface"
export TORCH_HOME="/data/horse/ws/jihe529c-main-rag/cache/torch"
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Set AWS credentials as environment variables
export AWS_ACCESS_KEY_ID=AKIAQUFLP6N3SLGG5ENI
export AWS_SECRET_ACCESS_KEY=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
export AWS_REGION=us-east-1
export AWS_PROFILE=sigir-participant

# Create cache directories
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p logs
mkdir -p "results/${OUTPUT_PREFIX}"

module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2

source env/bin/activate

# Run parameters
MODEL="tiiuae/Falcon3-10B-Instruct"
N_VALUE=0.5
ALPHA=0.7
TOP_K=20
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run MAIN-RAG with hybrid retriever
srun python test_RAG.py \
    --model $MODEL \
    --n $N_VALUE \
    --alpha $ALPHA \
    --top_k $TOP_K \
    --data_file $SPLIT_FILE \
    --output_format jsonl \
    --output_dir "results/${OUTPUT_PREFIX}" \
    > "logs/main_rag_${OUTPUT_PREFIX}_${TIMESTAMP}.log" 2>&1

echo "Job for $SPLIT_FILE completed"
