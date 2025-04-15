#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=main-rag-liverag
#SBATCH --output=logs/main_rag_%j.log

# Set cache directories
export HF_DATASETS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_datasets"
export TRANSFORMERS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_models"
export HF_HOME="/data/horse/ws/jihe529c-main-rag/cache/huggingface"
export TORCH_HOME="/data/horse/ws/jihe529c-main-rag/cache/torch"
export HF_DATASETS_TRUST_REMOTE_CODE=1

# API Keys (best practice would be to store these more securely)
export PINECONE_API_KEY="pcsk_6yNXRS_K6NZ1qRf9gQht23BkagEU2jr4EZwVtTScvfpgQ1DfS5P34RJKkNcJEMxwNqUygD"
export FALCON_API_KEY="ai71-api-2f336f7f-318c-4773-bee9-a8651428640b"

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

# Install dependencies if not already installed
pip install -q pinecone-client==5.4.2 sentence-transformers==2.7.0 requests==2.31.0

# Parameters
N_VALUE=0.5                  # Adaptive judge bar parameter
QA_FILE="datamorgana_qa_pairs.json"  # Path to your DataMorgana QA pairs

# Check that API keys have been properly set
if [ "$PINECONE_API_KEY" = "your_pinecone_api_key_here" ]; then
    echo "Error: Please set your actual Pinecone API key in the script"
    exit 1
fi

if [ "$FALCON_API_KEY" = "your_ai71_api_key_here" ]; then
    echo "Error: Please set your actual Falcon API key in the script"
    exit 1
fi

# Run evaluation on DataMorgana questions
echo "Running MAIN-RAG evaluation with Pinecone and Falcon..."
srun python run_main_rag_integration.py \
    --pinecone_key $PINECONE_API_KEY \
    --falcon_key $FALCON_API_KEY \
    --n $N_VALUE \
    --qa_file $QA_FILE

echo "Job completed"
