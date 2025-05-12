from datasets import load_dataset
import os
import json

class BenchmarkDatasets:
    def __init__(self, data_dir="data/benchmarks"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_arc_challenge(self):
        """Load ARC-Challenge dataset."""
        with open(os.path.join(self.data_dir, "arc_challenge/test.json"), 'r') as f:
            return json.load(f)
        
    def load_triviaqa(self):
        """Load TriviaQA dataset."""
        with open(os.path.join(self.data_dir, "triviaqa/validation.json"), 'r') as f:
            return json.load(f)
        
    def load_popqa(self):
        """Load PopQA dataset."""
        with open(os.path.join(self.data_dir, "popqa/test.json"), 'r') as f:
            return json.load(f)
        
    def load_asqa(self):
        """Load ASQA dataset."""
        with open(os.path.join(self.data_dir, "asqa/dev.json"), 'r') as f:
            return json.load(f)
