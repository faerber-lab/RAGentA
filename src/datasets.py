from datasets import load_dataset
import os
import json

class BenchmarkDatasets:
    def __init__(self, data_dir="data/benchmarks"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_arc_challenge(self):
        """Load ARC-Challenge dataset."""
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
        return [{
            "question": item["question"],
            "choices": item["choices"]["text"],
            "answer_idx": item["choices"]["label"].index(item["answerKey"]),
            "answer": item["choices"]["text"][item["choices"]["label"].index(item["answerKey"])]
        } for item in dataset]
    
    def load_triviaqa(self):
        """Load TriviaQA dataset (using validation set since test is not publicly available)."""
        # For actual paper reproduction, you'd need the specific test split from Asai et al.
        dataset = load_dataset("trivia_qa", "unfiltered", split="validation[:1000]")  # Limit for testing
        return [{
            "question": item["question"],
            "answer": item["answer"]["value"]
        } for item in dataset]
    
    def load_popqa(self):
        """Load PopQA dataset (long-tail subset)."""
        # This is a placeholder - you'd need to download the actual PopQA dataset
        # For demonstration purposes, we'll create a small mock dataset
        if not os.path.exists(os.path.join(self.data_dir, "popqa.json")):
            print("PopQA dataset not found. Creating mock data. For actual benchmarking, download the real dataset.")
            mock_data = [
                {"question": "Who is Montxu Miranda?", "answer": "Spanish pole vaulter"},
                {"question": "What is the capital of Gmina Czorsztyn?", "answer": "Maniowy"},
                # Add more examples based on the paper's case studies
            ]
            with open(os.path.join(self.data_dir, "popqa.json"), 'w') as f:
                json.dump(mock_data, f)
        
        with open(os.path.join(self.data_dir, "popqa.json"), 'r') as f:
            return json.load(f)
    
    def load_asqa(self):
        """Load ASQA dataset."""
        # This is a placeholder - you'd need to download the actual ASQA dataset
        # For demonstration purposes, we'll create a small mock dataset
        if not os.path.exists(os.path.join(self.data_dir, "asqa.json")):
            print("ASQA dataset not found. Creating mock data. For actual benchmarking, download the real dataset.")
            mock_data = [
                {"question": "What are the effects of climate change?", "answer": "Rising temperatures, sea level rise, and extreme weather events."},
                {"question": "How does a neural network work?", "answer": "Neural networks process data through interconnected layers of nodes, applying weights and activation functions to transform inputs into outputs."},
                # Add more examples
            ]
            with open(os.path.join(self.data_dir, "asqa.json"), 'w') as f:
                json.dump(mock_data, f)
        
        with open(os.path.join(self.data_dir, "asqa.json"), 'r') as f:
            return json.load(f)
