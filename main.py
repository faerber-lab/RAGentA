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

import argparse
import sys
import json
from datetime import datetime
from src.retriever import WikipediaRetriever
from src.main_rag import MAIN_RAG
from src.datasets import BenchmarkDatasets
from src.evaluation import exact_match_accuracy, contains_answer, calculate_rouge, choice_accuracy
from tqdm import tqdm

def run_benchmark(benchmark_name, dataset, rag_system, results_dir="results/benchmarks"):
    """Run benchmark on a dataset."""
    os.makedirs(results_dir, exist_ok=True)

    # For ARC-Challenge
    dataset = dataset_loader.load_arc_challenge()
    
    predictions = []
    references = []
    details = []
    
    print(f"Running benchmark on {benchmark_name} with {len(dataset)} examples...")
    
    for item in tqdm(dataset):
        query = item["question"]
        reference = item["answer"]
        choices = item["choices"]
        
        # Process the query with choices
        answer, debug_info = rag_system.answer_query(query, choices=choices)
        
        predictions.append(answer)
        references.append(reference)
        
        # Save details for analysis
        details.append({
            "question": query,
            "reference": reference,
            "prediction": answer,
            "tau_q": debug_info["tau_q"],
            "filtered_count": len(debug_info["filtered_docs"])
        })
    
    # Calculate metrics based on benchmark
    metrics = {}
    if benchmark_name == "ARC-Challenge":
        metrics["accuracy"] = choice_accuracy(
            predictions, 
            [item["choices"] for item in dataset],
            [item["answer_idx"] for item in dataset]
        )
    elif benchmark_name in ["TriviaQA", "PopQA"]:
        metrics["contains_answer"] = contains_answer(predictions, references)
    elif benchmark_name == "ASQA":
        metrics["exact_match"] = exact_match_accuracy(predictions, references)
        metrics["rouge"] = calculate_rouge(predictions, references)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{benchmark_name}_{timestamp}.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "benchmark": benchmark_name,
            "metrics": metrics,
            "details": details
        }, f, indent=2)
    
    print(f"Benchmark {benchmark_name} results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    print(f"Results saved to {results_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='MAIN-RAG: Multi-Agent Filtering RAG Benchmarks')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1', help='Model to use')
    parser.add_argument('--n', type=float, default=0.0, help='Adjustment factor for adaptive judge bar')
    parser.add_argument('--benchmark', type=str, default='all', help='Benchmark to run (arc, triviaqa, popqa, asqa, or all)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples per benchmark')
    args = parser.parse_args()
    
    # Initialize the retriever
    retriever = WikipediaRetriever()
    
    # Initialize MAIN-RAG
    rag_system = MAIN_RAG(retriever, agent_model=args.model, n=args.n)
    
    # Load benchmark datasets
    dataset_loader = BenchmarkDatasets()
    
    # Run benchmarks
    all_metrics = {}
    
    if args.benchmark in ['arc', 'all']:
        dataset = dataset_loader.load_arc_challenge()
        if args.limit:
            dataset = dataset[:args.limit]
        all_metrics["ARC-Challenge"] = run_benchmark("ARC-Challenge", dataset, rag_system)
    
    if args.benchmark in ['triviaqa', 'all']:
        dataset = dataset_loader.load_triviaqa()
        if args.limit:
            dataset = dataset[:args.limit]
        all_metrics["TriviaQA"] = run_benchmark("TriviaQA", dataset, rag_system)
    
    if args.benchmark in ['popqa', 'all']:
        dataset = dataset_loader.load_popqa()
        if args.limit:
            dataset = dataset[:args.limit]
        all_metrics["PopQA"] = run_benchmark("PopQA", dataset, rag_system)
    
    if args.benchmark in ['asqa', 'all']:
        dataset = dataset_loader.load_asqa()  # This will now load from asqa/dev.json
        if args.limit:
            dataset = dataset[:args.limit]
        all_metrics["ASQA"] = run_benchmark("ASQA", dataset, rag_system)
    
    # Print summary
    print("\n=== BENCHMARK SUMMARY ===")
    for benchmark, metrics in all_metrics.items():
        print(f"{benchmark}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()
