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

import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from rouge import Rouge


class BenchmarkDatasets:
    def __init__(self, data_dir="data/benchmarks"):
        """
        Helper class to load benchmark datasets.
        
        Args:
            data_dir: Directory containing benchmark data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_arc_challenge(self):
        """Load ARC-Challenge dataset."""
        arc_path = os.path.join(self.data_dir, "arc_challenge/test.json")
        if not os.path.exists(arc_path):
            raise FileNotFoundError(f"ARC-Challenge dataset not found at {arc_path}. Please download it first.")
            
        with open(arc_path, "r") as f:
            return json.load(f)

    def load_triviaqa(self):
        """Load TriviaQA dataset."""
        triviaqa_path = os.path.join(self.data_dir, "triviaqa/validation.json")
        if not os.path.exists(triviaqa_path):
            raise FileNotFoundError(f"TriviaQA dataset not found at {triviaqa_path}. Please download it first.")
            
        with open(triviaqa_path, "r") as f:
            return json.load(f)

    def load_popqa(self):
        """Load PopQA dataset."""
        popqa_path = os.path.join(self.data_dir, "popqa/test.json")
        if not os.path.exists(popqa_path):
            raise FileNotFoundError(f"PopQA dataset not found at {popqa_path}. Please download it first.")
            
        with open(popqa_path, "r") as f:
            return json.load(f)

    def load_asqa(self):
        """Load ASQA dataset."""
        asqa_path = os.path.join(self.data_dir, "asqa/dev.json")
        if not os.path.exists(asqa_path):
            raise FileNotFoundError(f"ASQA dataset not found at {asqa_path}. Please download it first.")
            
        with open(asqa_path, "r") as f:
            return json.load(f)


# Evaluation metrics used in the paper
def exact_match_accuracy(predictions, references):
    """Calculate exact match accuracy."""
    correct = 0
    for pred, ref in zip(predictions, references):
        if pred.lower() == ref.lower():
            correct += 1
    return correct / len(predictions)


def contains_answer(predictions, references):
    """Check if reference is contained in prediction."""
    correct = 0
    for pred, ref in zip(predictions, references):
        if ref.lower() in pred.lower():
            correct += 1
    return correct / len(predictions)


def calculate_rouge(predictions, references):
    """Calculate ROUGE scores with safeguards for empty predictions."""
    # Ensure no empty predictions
    valid_predictions = []
    valid_references = []

    for pred, ref in zip(predictions, references):
        # Replace empty predictions with a placeholder
        if not pred or pred.strip() == "":
            pred = "no answer provided"
        valid_predictions.append(pred)
        valid_references.append(ref)

    try:
        rouge = Rouge()
        scores = rouge.get_scores(valid_predictions, valid_references, avg=True)
        return scores
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        # Return dummy scores
        return {
            "rouge-1": {"r": 0.0, "p": 0.0, "f": 0.0},
            "rouge-2": {"r": 0.0, "p": 0.0, "f": 0.0},
            "rouge-l": {"r": 0.0, "p": 0.0, "f": 0.0},
        }


def choice_accuracy(predictions, choices_list, answer_indices):
    """Calculate accuracy for multiple-choice questions."""
    correct = 0
    for pred, choices, answer_idx in zip(predictions, choices_list, answer_indices):
        # Extract just the letter from the prediction
        pred = pred.strip().upper()
        if pred and pred[0] in "ABCD":
            selected_idx = ord(pred[0]) - ord("A")
            if selected_idx == answer_idx:
                correct += 1

    return correct / len(predictions)


def calculate_mauve(predictions, references):
    """
    Calculate MAUVE score - a metric for evaluating text generation quality.
    This is an optional metric used for ASQA in the paper.
    
    Note: This requires installing mauve-text package
    """
    try:
        import mauve
        out = mauve.compute_mauve(p_text=predictions, q_text=references, device_id=0, max_text_length=512)
        return out.mauve
    except ImportError:
        print("Warning: mauve-text package not installed. Skipping MAUVE calculation.")
        return 0.0
    except Exception as e:
        print(f"Error calculating MAUVE: {e}")
        return 0.0


def run_benchmark(
    benchmark_name, dataset, rag_system, n_value=0.0, results_dir="results/benchmarks"
):
    """Run benchmark on a dataset."""
    os.makedirs(results_dir, exist_ok=True)

    # Update the n value
    rag_system.n = n_value
    print(f"Using adaptive judge bar adjustment n={n_value}")

    predictions = []
    references = []
    details = []

    print(f"Running benchmark on {benchmark_name} with {len(dataset)} examples...")

    for item in tqdm(dataset):
        query = item["question"]
        reference = item["answer"]
        choices = item.get("choices", None)  # For ARC-Challenge

        try:
            # Process the query
            answer, debug_info = rag_system.answer_query(query, choices=choices)

            # Ensure answer is never None or empty
            if answer is None or answer.strip() == "":
                answer = "no answer provided"

            predictions.append(answer)
            references.append(reference)

            # Save details for analysis
            details.append(
                {
                    "question": query,
                    "reference": reference,
                    "prediction": answer,
                    "tau_q": debug_info["tau_q"],
                    "filtered_count": len(debug_info["filtered_docs"]),
                }
            )
        except Exception as e:
            print(f"Error processing query: {query}")
            print(f"Error: {e}")
            # Add a placeholder to maintain alignment
            predictions.append("error occurred")
            references.append(reference)

    # Calculate metrics based on benchmark
    metrics = {}
    if benchmark_name == "ARC-Challenge":
        metrics["accuracy"] = choice_accuracy(
            predictions,
            [item["choices"] for item in dataset],
            [item["answer_idx"] for item in dataset],
        )
    elif benchmark_name in ["TriviaQA", "PopQA"]:
        metrics["contains_answer"] = contains_answer(predictions, references)
    elif benchmark_name == "ASQA":
        # Handle potential empty predictions for ROUGE calculation
        metrics["exact_match"] = exact_match_accuracy(predictions, references)
        metrics["rouge"] = calculate_rouge(predictions, references)
        # Calculate MAUVE if available
        try:
            metrics["mauve"] = calculate_mauve(predictions, references)
        except:
            # Skip MAUVE if it fails
            pass

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{benchmark_name}_{timestamp}.json")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {"benchmark": benchmark_name, "metrics": metrics, "details": details},
            f,
            indent=2,
        )

    print(f"Benchmark {benchmark_name} results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    print(f"Results saved to {results_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="MAIN-RAG: Multi-Agent Filtering RAG Evaluation")
    parser.add_argument(
        "--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Model to use"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        help="Benchmark to run (arc, triviaqa, popqa, asqa, or all)",
    )
    parser.add_argument(
        "--n", type=float, default=0.0, help="Adjustment factor for adaptive judge bar"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of examples per benchmark"
    )
    parser.add_argument(
        "--retriever_model", type=str, default="facebook/contriever-msmarco", 
        help="Model for retriever embeddings"
    )
    args = parser.parse_args()

    # Default n values based on empirical tuning (start with all at 0.0 as in the paper)
    default_n_values = {"arc": 0.0, "triviaqa": 0.0, "popqa": 0.0, "asqa": 0.0}

    # Initialize the retriever
    print("Initializing retriever...")
    from wikipedia_retriever import WikipediaRetriever
    retriever = WikipediaRetriever(model_name=args.retriever_model)

    # Load dataset loader
    dataset_loader = BenchmarkDatasets()

    # Import and initialize MAIN_RAG
    from main_rag import MAIN_RAG

    # Run benchmarks
    all_metrics = {}

    if args.benchmark == "all":
        benchmarks = ["arc", "triviaqa", "popqa", "asqa"]
    else:
        benchmarks = [args.benchmark]

    for benchmark in benchmarks:
        # Use provided n or default value for this benchmark
        n_value = args.n if args.n is not None else default_n_values[benchmark]

        print(f"\n=== Running benchmark: {benchmark.upper()} with n={n_value} ===\n")

        # Initialize MAIN-RAG system with appropriate n value
        rag_system = MAIN_RAG(retriever, agent_model=args.model, n=n_value)

        # Load appropriate dataset
        if benchmark == "arc":
            dataset = dataset_loader.load_arc_challenge()
            if args.limit:
                dataset = dataset[:args.limit]
            all_metrics["ARC-Challenge"] = run_benchmark(
                "ARC-Challenge", dataset, rag_system, n_value=n_value
            )

        elif benchmark == "triviaqa":
            dataset = dataset_loader.load_triviaqa()
            if args.limit:
                dataset = dataset[:args.limit]
            all_metrics["TriviaQA"] = run_benchmark("TriviaQA", dataset, rag_system, n_value=n_value)

        elif benchmark == "popqa":
            dataset = dataset_loader.load_popqa()
            if args.limit:
                dataset = dataset[:args.limit]
            all_metrics["PopQA"] = run_benchmark("PopQA", dataset, rag_system, n_value=n_value)

        elif benchmark == "asqa":
            dataset = dataset_loader.load_asqa()
            if args.limit:
                dataset = dataset[:args.limit]
            all_metrics["ASQA"] = run_benchmark("ASQA", dataset, rag_system, n_value=n_value)

    # Print summary of all results
    print("\n=== BENCHMARK SUMMARY ===")
    for benchmark, metrics in all_metrics.items():
        print(f"{benchmark}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
