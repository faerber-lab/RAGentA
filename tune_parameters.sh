#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=main-rag-tune

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

# Create directories
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p logs
mkdir -p results

module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2

source env/bin/activate

# Install required dependencies
pip install pinecone-client==5.4.2 boto3==1.35.88 opensearch-py==2.8.0 sentence-transformers tqdm

# Parameters
MODEL="tiiuae/falcon-3-10b-instruct"
TOP_K=20
DATA_FILE="datamorgana_questions.json"

# Test different combinations of n and alpha
for N_VALUE in 0.0 0.5 1.0 1.5
do
  for ALPHA in 0.5 0.7 0.9
  do
    echo "Running with n=$N_VALUE, alpha=$ALPHA"
    srun python main_rag_runner.py \
        --model $MODEL \
        --n $N_VALUE \
        --alpha $ALPHA \
        --top_k $TOP_K \
        --data_file $DATA_FILE \
        > logs/main_rag_hybrid_n${N_VALUE}_a${ALPHA}.log 2>&1
  done
done

echo "Parameter tuning completed"