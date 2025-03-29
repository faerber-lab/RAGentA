import os
import sys
import json
from datasets import load_dataset
from tqdm import tqdm

# Get the current username
# username = os.getenv('USER')

# Set cache directories on horse workspace
cache_base = f"/data/horse/ws/jihe529c-main-rag/cache"
os.makedirs(cache_base, exist_ok=True)

# Set all cache directories
os.environ["HF_DATASETS_CACHE"] = f"{cache_base}/hf_datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{cache_base}/hf_models"
os.environ["HF_HOME"] = f"{cache_base}/huggingface"
os.environ["TORCH_HOME"] = f"{cache_base}/torch"
# Accept running custom code for datasets
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

# Create cache directories
for cache_dir in [
    os.environ["HF_DATASETS_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["HF_HOME"],
    os.environ["TORCH_HOME"]
]:
    os.makedirs(cache_dir, exist_ok=True)


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
        # Convert possible_answers from string to list
        answers = []
        if "possible_answers" in item:
            try:
                # Handle when possible_answers is a string representation of a list
                import json
                answers = json.loads(item["possible_answers"].replace("'", '"'))
            except:
                # If JSON parsing fails, try eval (safer than direct eval)
                try:
                    import ast
                    answers = ast.literal_eval(item["possible_answers"])
                except:
                    # Fall back to just using the string
                    answers = [item["possible_answers"]]
        
        # If the question asks for occupation, we can also check the 'obj' field
        if 'occupation' in item['question'].lower() and 'obj' in item:
            if not answers:  # Only use obj if answers list is empty
                answers = [item['obj']]
        
        formatted_data.append({
            "question": item["question"],
            "answer": answers[0] if answers else "",
            "alternative_answers": answers
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
    
    formatted_data = []
    for item in tqdm(dataset):
        # Extract the main ambiguous question
        question = item["ambiguous_question"]
        
        # Extract the long answer from annotations if available
        answer = ""
        if "annotations" in item and len(item["annotations"]) > 0:
            for annotation in item["annotations"]:
                if "long_answer" in annotation and annotation["long_answer"]:
                    answer = annotation["long_answer"]
                    break
        
        # If no long answer, try to construct one from qa_pairs
        if not answer and "qa_pairs" in item and len(item["qa_pairs"]) > 0:
            answers = []
            for qa_pair in item["qa_pairs"]:
                if "short_answers" in qa_pair and qa_pair["short_answers"]:
                    qa_question = qa_pair.get("question", "")
                    qa_answer = qa_pair["short_answers"][0]
                    answers.append(f"{qa_question}: {qa_answer}")
            
            if answers:
                answer = " ".join(answers)
        
        formatted_data.append({
            "question": question,
            "answer": answer
        })
    
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
