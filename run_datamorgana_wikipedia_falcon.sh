#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=falcon-wiki
#SBATCH --output=logs/falcon_wiki_%j.log

# Set cache directories
export HF_DATASETS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_datasets"
export TRANSFORMERS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_models"
export HF_HOME="/data/horse/ws/jihe529c-main-rag/cache/huggingface"
export TORCH_HOME="/data/horse/ws/jihe529c-main-rag/cache/torch"
export HF_DATASETS_TRUST_REMOTE_CODE=1

# API Key for Falcon (replace with your actual key)
export FALCON_API_KEY="your_ai71_api_key_here"

# Parameters
QA_FILE="datamorgana_qa_pairs.json"  # Path to your DataMorgana QA pairs
N_VALUE=0.5                         # Adaptive judge bar parameter

# Create directories
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p logs
mkdir -p results

# Load modules
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2

# Activate virtual environment
source env/bin/activate

# Check if API key is set
if [ "$FALCON_API_KEY" = "your_ai71_api_key_here" ]; then
    echo "Error: Please set your actual Falcon API key in the script"
    exit 1
fi

# Run evaluation
echo "Evaluating DataMorgana questions using Falcon with Wikipedia retriever..."
srun python evaluate_datamorgana_with_wikipedia.py \
    --falcon_key $FALCON_API_KEY \
    --qa_file $QA_FILE \
    --n $N_VALUE

echo "Job completed"
