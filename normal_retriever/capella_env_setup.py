#!/usr/bin/env python
"""
Environment setup for MAIN-RAG on Capella cluster
Include this at the beginning of all Python scripts
"""

import os

# Set cache directories for HuggingFace and PyTorch
os.environ["HF_DATASETS_CACHE"] = "/data/horse/ws/jihe529c-main-rag/cache/hf_datasets"
os.environ["TRANSFORMERS_CACHE"] = "/data/horse/ws/jihe529c-main-rag/cache/hf_models"
os.environ["HF_HOME"] = "/data/horse/ws/jihe529c-main-rag/cache/huggingface"
os.environ["TORCH_HOME"] = "/data/horse/ws/jihe529c-main-rag/cache/torch"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

# Create directories if they don't exist
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

# Set PyTorch CUDA specific settings for optimal H100 performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Import this before any other deep learning imports
def setup_torch_for_h100():
    """Configure PyTorch for optimal H100 performance"""
    import torch
    
    # Enable TF32 precision (good balance of speed and accuracy for H100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set default device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU by default
        
        # Print GPU info
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Warning: CUDA not available, using CPU")
    
    return torch.cuda.is_available()

# Note: Call setup_torch_for_h100() after importing torch in your scripts
