#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=main-rag-tune
#SBATCH --output=logs/tuning_%j.log

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
mkdir -p logs
mkdir -p results/tuning/small_subset
mkdir -p results/tuning/validation

module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5
module load PyTorch/2.1.2

source env/bin/activate

# Make sure matplotlib is installed
pip install matplotlib

# Parameters
BENCHMARK=${1:-"all"}   # Default to "all" if not provided
MODEL="mistralai/Mistral-7B-v0.1"
SUBSET_SIZE=50          # Small subset for initial tuning
VALIDATION_SIZE=200     # Larger subset for validation
N_VALUES="0.0,0.5,1.0,1.5"  # Values to test, as mentioned in the paper

echo "Starting parameter tuning for benchmark: $BENCHMARK"
echo "Using model: $MODEL"
echo "Initial subset size: $SUBSET_SIZE"
echo "Validation size: $VALIDATION_SIZE"
echo "Testing n values: $N_VALUES"

# Run the parameter tuning script
srun python parameter_tuning.py \
    --model $MODEL \
    --benchmark $BENCHMARK \
    --subset_size $SUBSET_SIZE \
    --validation_size $VALIDATION_SIZE \
    --n_values $N_VALUES

echo "Parameter tuning completed"

# After finding optimal n values, you can run the full benchmark with the best n
# This section is commented out as you might want to run it in a separate job
# after reviewing the tuning results

: <<'COMMENT'
# Get optimal n value from the results (you would need to implement this)
OPTIMAL_N_FILE=$(ls -t results/tuning/optimal_n_values_*.json | head -n1)
OPTIMAL_N=$(python -c "import json; f=open('$OPTIMAL_N_FILE'); data=json.load(f); print(data.get('$BENCHMARK', 0.0))")

echo "Running full benchmark with optimal n value: $OPTIMAL_N"

# Run the full benchmark with the optimal n value
srun python benchmark_evaluation.py \
    --model $MODEL \
    --benchmark $BENCHMARK \
    --n $OPTIMAL_N
COMMENT

echo "Job completed"
