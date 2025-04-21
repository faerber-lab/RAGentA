#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=main-rag-hybrid

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

# Create AWS credentials directory and files - only if they don't exist
if [ ! -d "$HOME/.aws" ]; then
  mkdir -p $HOME/.aws
  cat > $HOME/.aws/credentials << EOL
[sigir-participant]
aws_access_key_id = ${AWS_ACCESS_KEY_ID}
aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}
EOL

  cat > $HOME/.aws/config << EOL
[profile sigir-participant]
region = ${AWS_REGION}
output = json
EOL

  # Set proper permissions
  chmod 600 $HOME/.aws/credentials $HOME/.aws/config
  
  echo "Created AWS credentials files in $HOME/.aws"
else
  echo "AWS credentials directory already exists"
fi

# Create cache directories
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p logs
mkdir -p results

# Load modules
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2

source env/bin/activate

# Install required dependencies
pip install pinecone-client==5.4.2 boto3==1.35.88 opensearch-py==2.8.0 sentence-transformers tqdm

# Run parameters
MODEL="tiiuae/falcon-3-10b-instruct"
N_VALUE=0.5
ALPHA=0.7
TOP_K=20
DATA_FILE="datamorgana_questions.json"

# Print debug information
echo "AWS credentials location: $HOME/.aws"
echo "AWS Profile: $AWS_PROFILE"
echo "AWS Region: $AWS_REGION"
echo "Running from directory: $(pwd)"
echo "Files in current directory: $(ls -la)"

# Run MAIN-RAG with hybrid retriever
srun python main_rag_runner.py \
    --model $MODEL \
    --n $N_VALUE \
    --alpha $ALPHA \
    --top_k $TOP_K \
    --data_file $DATA_FILE \
    > logs/main_rag_hybrid_n${N_VALUE}_a${ALPHA}.log 2>&1

echo "Job completed"