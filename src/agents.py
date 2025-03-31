import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

class LLMAgent:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", device="cuda"):
        """
        LLM agent for generating text and calculating probabilities.
        
        Args:
            model_name: Hugging Face model name
            device: Device to use (cuda or cpu)
        """
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Optimizations for H100 GPUs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,  # bfloat16 is well-supported on H100s
            device_map="auto",
        )
        self.device = device
        print("Model loaded")
        
    def generate(self, prompt, max_new_tokens=256, temperature=0.7, repetition_penalty=1.0):
        """Generate text based on prompt with temperature control."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], 
                                         skip_special_tokens=True)
        return response
        
    def get_log_probs(self, prompt, target_tokens=["Yes", "No"]):
        """Calculate log probabilities for specific tokens."""
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
            target_ids.append(token_ids[0] if token_ids else self.tokenizer.unk_token_id)
        
        # Calculate log probabilities using softmax
        log_probs = torch.log_softmax(logits, dim=0)
        target_log_probs = {
            token: log_probs[tid].item() 
            for token, tid in zip(target_tokens, target_ids)
        }
        
        return target_log_probs
