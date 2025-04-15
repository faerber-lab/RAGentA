import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalFalconAgent:
    """
    LLM agent that uses Falcon3-10B-Instruct directly on the local system.
    
    This agent implements the same interface as FalconAgent but loads and runs
    the model directly instead of making API calls.
    """
    
    def __init__(self, model_name="tiiuae/falcon-3-10b-instruct", device="cuda", precision="bfloat16"):
        """
        Initialize the Falcon agent with a local model.
        
        Args:
            model_name: Model identifier on Hugging Face
            device: Device to use (cuda or cpu)
            precision: Model precision (bfloat16, float16, or float32)
        """
        # Retrieve the Hugging Face token from the environment
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable is not set.")

        # Initialize the tokenizer with the authentication token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        
        # Determine torch dtype based on precision
        if precision == "bfloat16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print("Using bfloat16 precision")
        elif precision == "float16":
            torch_dtype = torch.float16
            print("Using float16 precision")
        else:
            torch_dtype = torch.float32
            print("Using float32 precision")
            
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",     # Automatically optimize GPU usage
            trust_remote_code=True # Required for some models
        )
        
        # Save device information
        self.device = self.model.device
        print(f"Model loaded on {self.device}")
    
    def generate(self, prompt, max_new_tokens=256):
        """
        Generate text using the local Falcon model.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text as a string
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Use greedy decoding as in MAIN-RAG
                pad_token_id=self.tokenizer.eos_token_id
            )
                
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        return response
    
    def get_log_probs(self, prompt, target_tokens=["Yes", "No"]):
        """
        Calculate log probabilities for specific tokens.
        
        Args:
            prompt: The input prompt
            target_tokens: List of tokens to get probabilities for
            
        Returns:
            Dictionary mapping tokens to their log probabilities
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get logits for the last token position
        logits = outputs.logits[0, -1, :]

        # Get token IDs for target tokens
        target_ids = []
        for token in target_tokens:
            # Handle different tokenizer behaviors
            token_ids = self.tokenizer.encode(" " + token, add_special_tokens=False)
            # Use the first token if multiple tokens
            target_ids.append(
                token_ids[0] if token_ids else self.tokenizer.unk_token_id
            )

        # Calculate log probabilities using softmax
        log_probs = torch.log_softmax(logits, dim=0)
        target_log_probs = {
            token: log_probs[tid].item()
            for token, tid in zip(target_tokens, target_ids)
        }

        return target_log_probs
    
    def batch_process(self, prompts, generate=True, max_new_tokens=256):
        """
        Process a batch of prompts in parallel.
        
        Args:
            prompts: List of prompt strings
            generate: If True, generate text; if False, return log probs for Yes/No
            max_new_tokens: Maximum new tokens for generation
            
        Returns:
            List of responses or log probs
        """
        if not prompts:
            return []
            
        results = []
        
        if generate:
            # Generate text for each prompt
            for prompt in prompts:
                results.append(self.generate(prompt, max_new_tokens))
        else:
            # Get log probs for each prompt
            for prompt in prompts:
                results.append(self.get_log_probs(prompt, ["Yes", "No"]))
                
        return results
