import os
import json
from datasets import load_dataset
from tqdm import tqdm

def prepare_arc_challenge():
    """Prepare ARC-Challenge dataset"""
    output_file = "data/benchmarks/arc_challenge/test.json"
    
    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"ARC-Challenge dataset already exists at {output_file}")
        return
    
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
    
    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Saved {len(formatted_data)} ARC-Challenge test examples")

def prepare_triviaqa():
    """Prepare TriviaQA dataset"""
    output_file = "data/benchmarks/triviaqa/validation.json"
    
    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"TriviaQA dataset already exists at {output_file}")
        return
    
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
    
    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Saved {len(formatted_data)} TriviaQA validation examples")

def prepare_popqa():
    """Prepare PopQA dataset"""
    output_file = "data/benchmarks/popqa/test.json"
    
    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"PopQA dataset already exists at {output_file}")
        return
    
    print("Preparing PopQA dataset...")
    os.makedirs("data/benchmarks/popqa", exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("akariasai/PopQA", split="test")
    
    formatted_data = []
    for item in tqdm(dataset):
        # Using possible_answers field which contains the list of acceptable answers
        if "possible_answers" in item:
            # Convert from string representation to actual list if needed
            if isinstance(item["possible_answers"], str):
                try:
                    possible_answers = eval(item["possible_answers"])  # Safely evaluate the string to a list
                except:
                    possible_answers = [item["possible_answers"]]
            else:
                possible_answers = item["possible_answers"]
                
            # Take the first answer as primary, keep all as alternatives
            formatted_data.append({
                "question": item["question"],
                "answer": possible_answers[0] if possible_answers else "",
                "alternative_answers": possible_answers
            })
        else:
            # Fallback if structure is different
            formatted_data.append({
                "question": item["question"],
                "answer": item.get("obj", ""),  # Use 'obj' field as fallback
                "alternative_answers": []
            })
    
    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Saved {len(formatted_data)} PopQA test examples")

def prepare_asqa():
    """Prepare ASQA dataset"""
    output_file = "data/benchmarks/asqa/dev.json"
    
    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"ASQA dataset already exists at {output_file}")
        return
    
    print("Preparing ASQA dataset...")
    os.makedirs("data/benchmarks/asqa", exist_ok=True)
    
    # Use "dev" split instead of "test"
    dataset = load_dataset("din0s/asqa", split="dev")
    
    print("ASQA dataset example item structure:")
    if len(dataset) > 0:
        example_item = dataset[0]
        print("Keys:", list(example_item.keys()))
        print("Example:", example_item)
    
    formatted_data = []
    for item in tqdm(dataset):
        formatted_item = {
            "question": item["question"],
        }
        
        # Add answer field based on actual structure
        if "answer" in item:
            formatted_item["answer"] = item["answer"]
        else:
            # Check for alternative field names
            for field in ["long_answer", "short_answer", "gold_answer"]:
                if field in item:
                    formatted_item["answer"] = item[field]
                    break
            else:
                formatted_item["answer"] = ""
                print(f"Warning: No answer field found for question: {item['question']}")
        
        formatted_data.append(formatted_item)
    
    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Saved {len(formatted_data)} ASQA dev examples")
    
def prepare_all_datasets():
    """Prepare all benchmark datasets"""
    prepare_arc_challenge()
    prepare_triviaqa()
    prepare_popqa()
    prepare_asqa()
    print("All benchmark datasets prepared successfully!")

if __name__ == "__main__":
    prepare_all_datasets()
