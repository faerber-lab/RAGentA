#!/usr/bin/env python
"""
MAIN-RAG Parameter Tuning Script
Finds optimal n values for each benchmark by first testing on small subsets,
then validating the best values on larger datasets.
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import necessary modules
from main_rag import MAIN_RAG
from wikipedia_retriever import WikipediaRetriever
from benchmark_evaluation import BenchmarkDatasets, run_benchmark


def find_optimal_n(benchmark_name, dataset, rag_system, n_values, results_dir="results/tuning"):
    """
    Run benchmark with different n values and find the optimal one.
    
    Args:
        benchmark_name: Name of the benchmark
        dataset: Dataset to evaluate on
        rag_system: MAIN_RAG instance
        n_values: List of n values to try
        results_dir: Directory to save results
        
    Returns:
        best_n: The n value with the best performance
        results: Dictionary of results for each n value
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Select the appropriate metric based on the benchmark
    if benchmark_name == "ARC-Challenge":
        metric_key = "accuracy"
    elif benchmark_name in ["TriviaQA", "PopQA"]:
        metric_key = "contains_answer"
    elif benchmark_name == "ASQA":
        metric_key = "rouge-l"  # Use ROUGE-L F1 score for ASQA
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    results = {}
    
    for n in tqdm(n_values, desc=f"Testing n values for {benchmark_name}"):
        print(f"\nRunning {benchmark_name} with n={n}")
        try:
            metrics = run_benchmark(
                benchmark_name, dataset, rag_system, n_value=n, 
                results_dir=f"{results_dir}/{benchmark_name}"
            )
            
            # Extract the main metric value
            if benchmark_name == "ASQA" and "rouge" in metrics:
                # For ASQA, use ROUGE-L F1 score
                metric_value = metrics["rouge"]["rouge-l"]["f"]
            else:
                metric_value = metrics.get(metric_key, 0.0)
                
            results[n] = {
                "metrics": metrics,
                "main_metric": metric_value
            }
            print(f"n={n}, {metric_key}={metric_value:.4f}")
        except Exception as e:
            print(f"Error running {benchmark_name} with n={n}: {e}")
            results[n] = {"error": str(e), "main_metric": 0.0}
    
    # Find best n value based on the main metric
    best_n = max(results.keys(), key=lambda n: results[n]["main_metric"])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"{benchmark_name}_n_tuning_{timestamp}.json")
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": benchmark_name,
            "best_n": best_n,
            "best_metric": results[best_n]["main_metric"],
            "results": {str(n): results[n] for n in results}
        }, f, indent=2)
        
    # Plot results
    plot_n_values(benchmark_name, results, metric_key, results_dir)
    
    return best_n, results


def plot_n_values(benchmark_name, results, metric_key, results_dir):
    """Create a plot of metric values for different n values."""
    n_values = sorted(results.keys())
    metric_values = [results[n]["main_metric"] for n in n_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, metric_values, 'o-', linewidth=2, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('n value')
    plt.ylabel(f'{metric_key}')
    plt.title(f'{benchmark_name}: Performance vs n value')
    
    # Add value labels
    for i, (n, val) in enumerate(zip(n_values, metric_values)):
        plt.annotate(f'{val:.4f}', (n, val), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    # Highlight best value
    best_n = max(n_values, key=lambda n: results[n]["main_metric"])
    best_idx = n_values.index(best_n)
    plt.scatter([best_n], [metric_values[best_idx]], color='red', s=100, zorder=5)
    plt.annotate(f'Best: n={best_n}', (best_n, metric_values[best_idx]), 
                textcoords="offset points", xytext=(0,-20), ha='center', 
                color='red', weight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(results_dir, f"{benchmark_name}_n_tuning_{timestamp}.png")
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Plot saved to {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="MAIN-RAG Parameter Tuning")
    parser.add_argument(
        "--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Model to use for agents"
    )
    parser.add_argument(
        "--benchmark", type=str, default="all",
        help="Benchmark to tune (arc, triviaqa, popqa, asqa, or all)"
    )
    parser.add_argument(
        "--subset_size", type=int, default=50,
        help="Number of examples to use for initial tuning"
    )
    parser.add_argument(
        "--validation_size", type=int, default=200,
        help="Number of examples to use for validation (set to None for full dataset)"
    )
    parser.add_argument(
        "--n_values", type=str, default="0.0,0.5,1.0,1.5",
        help="Comma-separated list of n values to try"
    )
    parser.add_argument(
        "--validation_n_values", type=str, default=None,
        help="Comma-separated list of n values to try for validation (defaults to top 2 from initial tuning)"
    )
    args = parser.parse_args()
    
    # Parse n values
    n_values = [float(n) for n in args.n_values.split(",")]
    validation_n_values = None
    if args.validation_n_values:
        validation_n_values = [float(n) for n in args.validation_n_values.split(",")]
    
    # Initialize the retriever
    print("Initializing retriever...")
    retriever = WikipediaRetriever()
    
    # Load dataset loader
    dataset_loader = BenchmarkDatasets()
    
    # Determine which benchmarks to run
    if args.benchmark == "all":
        benchmarks = ["arc", "triviaqa", "popqa", "asqa"]
    else:
        benchmarks = [args.benchmark]
    
    # Initialize MAIN-RAG system
    print(f"Initializing MAIN-RAG with model {args.model}...")
    rag_system = MAIN_RAG(retriever, agent_model=args.model)
    
    # Dictionary to store optimal n values
    optimal_n_values = {}
    
    # Step 1: Find approximate optimal n values using small subsets
    print(f"\n=== STEP 1: Finding approximate optimal n values using {args.subset_size} examples ===")
    for benchmark in benchmarks:
        print(f"\n=== Tuning {benchmark.upper()} ===")
        
        # Load dataset
        if benchmark == "arc":
            dataset = dataset_loader.load_arc_challenge()
            dataset_small = dataset[:args.subset_size]
            benchmark_name = "ARC-Challenge"
        elif benchmark == "triviaqa":
            dataset = dataset_loader.load_triviaqa()
            dataset_small = dataset[:args.subset_size]
            benchmark_name = "TriviaQA"
        elif benchmark == "popqa":
            dataset = dataset_loader.load_popqa()
            dataset_small = dataset[:args.subset_size]
            benchmark_name = "PopQA"
        elif benchmark == "asqa":
            dataset = dataset_loader.load_asqa()
            dataset_small = dataset[:args.subset_size]
            benchmark_name = "ASQA"
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        
        # Find optimal n value for small subset
        best_n, results = find_optimal_n(
            benchmark_name, dataset_small, rag_system, n_values,
            results_dir="results/tuning/small_subset"
        )
        
        # Store optimal n value
        optimal_n_values[benchmark] = best_n
        
        # Select top 2 n values for validation
        if validation_n_values is None:
            sorted_n = sorted(results.keys(), key=lambda n: results[n]["main_metric"], reverse=True)
            validation_n_values_benchmark = sorted_n[:min(2, len(sorted_n))]
        else:
            validation_n_values_benchmark = validation_n_values
        
        # Step 2: Validate on larger dataset
        if args.validation_size:
            print(f"\n=== STEP 2: Validating top n values on {args.validation_size} examples ===")
            
            # Load validation dataset
            if args.validation_size is None or args.validation_size >= len(dataset):
                validation_dataset = dataset
                print(f"Using full dataset ({len(dataset)} examples)")
            else:
                validation_dataset = dataset[:args.validation_size]
                print(f"Using {args.validation_size} examples")
            
            # Validate top n values
            best_validation_n, _ = find_optimal_n(
                benchmark_name, validation_dataset, rag_system, validation_n_values_benchmark,
                results_dir="results/tuning/validation"
            )
            
            # Update optimal n value based on validation
            optimal_n_values[benchmark] = best_validation_n
    
    # Print summary of optimal n values
    print("\n=== OPTIMAL N VALUES ===")
    for benchmark, n in optimal_n_values.items():
        print(f"{benchmark.upper()}: n = {n}")
    
    # Save optimal n values
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimal_n_file = os.path.join("results/tuning", f"optimal_n_values_{timestamp}.json")
    
    with open(optimal_n_file, "w", encoding="utf-8") as f:
        json.dump(optimal_n_values, f, indent=2)
    
    print(f"\nOptimal n values saved to {optimal_n_file}")


if __name__ == "__main__":
    main()
