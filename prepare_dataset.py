import os
import json
from datasets import load_dataset
from tqdm import tqdm

def prepare_arc_challenge():
    """Prepare ARC-Challenge dataset"""
    print("Preparing ARC-Challenge dataset...")
    os.makedirs("data/benchmarks/arc_challenge", exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    
    formatted_data = []
    for item in tqdm(dataset):
        formatted_data.append({
            "question": item["question"],
            "choices": item["choices"]["text"],
            "answer_idx": item["choices"]["label"].index(item["answerKey"]),
            "answer": item["choices"]["text"][item["choices"]["label"].index(item["answerKey"])]
        })
    
    with open("data/benchmarks/arc_challenge/test.json", "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Saved {len(formatted_data)} ARC-Challenge test examples")

def prepare_triviaqa():
    """Prepare TriviaQA dataset"""
    print("Preparing TriviaQA dataset...")
    os.makedirs("data/benchmarks/triviaqa", exist_ok=True)
    
    # Load dataset - using validation as the test set is not publicly available
    dataset = load_dataset("trivia_qa", "unfiltered", split="validation")
    
    formatted_data = []
    for item in tqdm(dataset):
        formatted_data.append({
            "question": item["question"],
            "answer": item["answer"]["value"],
            "aliases": item["answer"]["aliases"]
        })
    
    with open("data/benchmarks/triviaqa/validation.json", "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Saved {len(formatted_data)} TriviaQA validation examples")

def prepare_popqa():
    """Prepare PopQA dataset"""
    print("Preparing PopQA dataset...")
    os.makedirs("data/benchmarks/popqa", exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("akariasai/PopQA", split="test")
    
    formatted_data = []
    for item in tqdm(dataset):
        formatted_data.append({
            "question": item["question"],
            "answer": item["answers"][0] if item["answers"] else "",  # Take first answer if available
            "all_answers": item["answers"] if "answers" in item else []
        })
    
    with open("data/benchmarks/popqa/test.json", "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Saved {len(formatted_data)} PopQA test examples")

def prepare_asqa():
    """Prepare ASQA dataset"""
    print("Preparing ASQA dataset...")
    os.makedirs("data/benchmarks/asqa", exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("din0s/asqa", split="test")
    
    formatted_data = []
    for item in tqdm(dataset):
        formatted_data.append({
            "question": item["question"],
            "answer": item["answer"],
            "long_answer": item["long_answer"] if "long_answer" in item else item["answer"]
        })
    
    with open("data/benchmarks/asqa/test.json", "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Saved {len(formatted_data)} ASQA test examples")

def prepare_all_datasets():
    """Prepare all benchmark datasets"""
    prepare_arc_challenge()
    prepare_triviaqa()
    prepare_popqa()
    prepare_asqa()
    print("All benchmark datasets prepared successfully!")

if __name__ == "__main__":
    prepare_all_datasets()
