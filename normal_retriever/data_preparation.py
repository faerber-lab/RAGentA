import os

# Set cache directories
os.environ["HF_DATASETS_CACHE"] = "/data/horse/ws/jihe529c-main-rag/cache/hf_datasets"
os.environ["TRANSFORMERS_CACHE"] = "/data/horse/ws/jihe529c-main-rag/cache/hf_models"
os.environ["HF_HOME"] = "/data/horse/ws/jihe529c-main-rag/cache/huggingface"
os.environ["TORCH_HOME"] = "/data/horse/ws/jihe529c-main-rag/cache/torch"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

# Create directories
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

import sys
import json
from datasets import load_dataset
from tqdm import tqdm


def prepare_arc_challenge():
    """
    Prepare ARC-Challenge dataset as used in the MAIN-RAG paper.
    Using the test split as is standard for evaluation.
    """
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
        formatted_data.append(
            {
                "question": item["question"],
                "choices": item["choices"]["text"],
                "answer_idx": item["choices"]["label"].index(item["answerKey"]),
                "answer": item["choices"]["text"][
                    item["choices"]["label"].index(item["answerKey"])
                ],
            }
        )

    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)

    print(f"Saved {len(formatted_data)} ARC-Challenge test examples")


def prepare_triviaqa():
    """
    Prepare TriviaQA dataset as used in the MAIN-RAG paper.
    The paper likely used the unfiltered version's validation set
    since the test set is not publicly available.
    """
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
        # Prepare primary answer and all possible aliases for flexible matching
        aliases = item["answer"]["aliases"]
        if item["answer"]["value"] not in aliases:
            aliases.append(item["answer"]["value"])
            
        formatted_data.append(
            {
                "question": item["question"],
                "answer": item["answer"]["value"],
                "aliases": aliases,
            }
        )

    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)

    print(f"Saved {len(formatted_data)} TriviaQA validation examples")


def prepare_popqa():
    """
    Prepare PopQA dataset as used in the MAIN-RAG paper.
    The paper specifically used the long-tail subset (entities with <100 monthly views).
    """
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
        # Following the paper, filter for long-tail entities (<100 monthly views)
        # Skip if 'popularity' field exists and is ≥ 100
        if "popularity" in item and item["popularity"] >= 100:
            continue
            
        # Convert possible_answers from string to list
        answers = []
        if "possible_answers" in item and item["possible_answers"]:
            try:
                # Handle when possible_answers is a string representation of a list
                import json
                answers = json.loads(item["possible_answers"].replace("'", '"'))
            except:
                # If JSON parsing fails, try ast.literal_eval (safer than direct eval)
                try:
                    import ast
                    answers = ast.literal_eval(item["possible_answers"])
                except:
                    # Fall back to just using the string
                    answers = [item["possible_answers"]]

        # If empty and question asks for occupation, check the 'obj' field
        if (not answers or (len(answers) == 1 and not answers[0])) and \
           "occupation" in item["question"].lower() and "obj" in item:
            answers = [item["obj"]]

        # Skip examples without answers
        if not answers or (len(answers) == 1 and not answers[0]):
            continue

        formatted_data.append(
            {
                "question": item["question"],
                "answer": answers[0],
                "alternative_answers": answers[1:] if len(answers) > 1 else [],
            }
        )

    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)

    print(f"Saved {len(formatted_data)} PopQA long-tail test examples")


def prepare_asqa():
    """
    Prepare ASQA dataset as used in the MAIN-RAG paper.
    This uses the dev set as is standard for evaluation.
    """
    output_file = "data/benchmarks/asqa/dev.json"

    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"ASQA dataset already exists at {output_file}")
        return

    print("Preparing ASQA dataset...")
    os.makedirs("data/benchmarks/asqa", exist_ok=True)

    # Use "dev" split as in the paper
    dataset = load_dataset("din0s/asqa", split="dev")

    formatted_data = []
    for item in tqdm(dataset):
        # Extract the ambiguous question (main query)
        question = item["ambiguous_question"]

        # Extract the reference answer (ground truth)
        # For ASQA, we want the long-form answer as used in the paper
        answer = ""
        if "annotations" in item and len(item["annotations"]) > 0:
            for annotation in item["annotations"]:
                if "long_answer" in annotation and annotation["long_answer"]:
                    answer = annotation["long_answer"]
                    break

        # If no long answer, construct a comprehensive answer from qa_pairs
        if not answer and "qa_pairs" in item and len(item["qa_pairs"]) > 0:
            answers = []
            for qa_pair in item["qa_pairs"]:
                if "short_answers" in qa_pair and qa_pair["short_answers"]:
                    qa_question = qa_pair.get("question", "")
                    qa_answer = qa_pair["short_answers"][0]
                    answers.append(f"{qa_question}: {qa_answer}")

            if answers:
                answer = " ".join(answers)

        # Skip examples without answers
        if not answer:
            continue

        formatted_data.append({"question": question, "answer": answer})

    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)

    print(f"Saved {len(formatted_data)} ASQA dev examples")


def prepare_all_datasets():
    """Prepare all benchmark datasets used in the MAIN-RAG paper"""
    prepare_arc_challenge()
    prepare_triviaqa()
    prepare_popqa()
    prepare_asqa()
    print("All benchmark datasets prepared successfully!")


if __name__ == "__main__":
    prepare_all_datasets()
