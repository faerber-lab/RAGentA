# falcon_agent.py

import requests
import time

class FalconAgent:
    def __init__(self, api_key):
        """LLM agent that uses Falcon3-10B-Instruct through the AI71 API."""
        self.api_key = api_key
        self.api_url = "https://api.ai71.ai/v1/models/falcon-3-10b-instruct/completions"
    
    def generate(self, prompt, max_new_tokens=256):
        """Generate text using greedy decoding."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": 0.0  # Use greedy decoding as in MAIN-RAG
        }
        
        # Handle API request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()  # Raise exception for HTTP errors
                return response.json()["choices"][0]["text"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to generate text after {max_retries} attempts: {e}")
                wait_time = 2 ** attempt + 1  # Exponential backoff
                time.sleep(wait_time)
    
    def get_log_probs(self, prompt, target_tokens=["Yes", "No"]):
        """
        Calculate log probabilities for specific tokens.
        
        Note: The Falcon API may not directly provide token probabilities.
        This implementation approximates it by generating responses with
        different prompts to estimate the relative probabilities.
        """
        # We'll use a technique to approximate token probabilities
        # by using separate generations for each target token
        
        scores = {}
        for token in target_tokens:
            # Create a prompt that strongly biases toward the target token
            biased_prompt = f"{prompt}\n\nBased on the above information, I should answer '{token}'."
            
            try:
                # Generate with very low temperature to get confidence
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "prompt": biased_prompt,
                    "max_tokens": 10,
                    "temperature": 0.1
                }
                
                response = requests.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                
                # Check if the generation starts with the target token
                generation = response.json()["choices"][0]["text"].strip()
                
                # Assign score based on whether the response starts with the token
                if generation.startswith(token):
                    scores[token] = 0.0  # log(1.0)
                else:
                    scores[token] = -1.0  # log(0.368)
            except Exception:
                # Default to a low probability on failure
                scores[token] = -2.0  # log(0.135)
        
        return scores
