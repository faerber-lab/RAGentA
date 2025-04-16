#!/usr/bin/env python
"""
Evaluate DataMorgana questions using Falcon with Wikipedia retriever.

This script loads QA pairs from a DataMorgana JSONL file, processes them 
using MAIN-RAG with a Wikipedia retriever and local Falcon LLM, and evaluates
the results.
"""

import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm

from main_rag import MAIN_RAG
from local_falcon_agent import LocalFalconAgent  # Import the local Falcon agent
from wikipedia_retriever import WikipediaRetriever


def load_jsonl(file_path):
    """
    Load JSONL file (each line is a separate JSON object).
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of JSON objects
    """
    qa_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    qa_pair = json.loads(line)
                    qa_pairs.append(qa_pair)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line[:100]}...")
                    print(f"Error details: {e}")
    return qa_pairs


def evaluate_datamorgana_with_local_falcon(datamorgana_file, falcon_model="tiiuae/Falcon3-10B-Instruct", n_value=0.5, output_file=None):
    """
    Evaluate DataMorgana questions using local Falcon with Wikipedia retriever.
    
    Args:
        datamorgana_file: Path to DataMorgana JSONL file
        falcon_model: Name or path of the Falcon model
        n_value: Adaptive judge bar parameter
        output_file: Path to save results
    """
    print(f"Loading DataMorgana questions from {datamorgana_file}...")
    
    # Load JSONL file
    qa_pairs = load_jsonl(datamorgana_file)
    print(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Initialize Wikipedia retriever
    print("Initializing Wikipedia retriever...")
    retriever = WikipediaRetriever()
    
    # Initialize local Falcon agent
    print(f"Initializing local Falcon model: {falcon_model}...")
    falcon_agent = LocalFalconAgent(model_name=falcon_model)
    
    # Initialize MAIN-RAG
    print(f"Initializing MAIN-RAG with n={n_value}...")
    rag_system = MAIN_RAG(
        retriever=retriever,
        agent_model=falcon_agent,  # Pass the pre-initialized agent
        n=n_value
    )
    
    # Process QA pairs
    results = []
    for i, qa_pair in enumerate(tqdm(qa_pairs, desc="Evaluating QA pairs")):
        question = qa_pair["question"]
        reference = qa_pair["answer"]
        
        print(f"\n[{i+1}/{len(qa_pairs)}] Processing question: {question}")
        
        try:
            # Generate answer using MAIN-RAG
            answer, debug_info = rag_system.answer_query(question)
            
            # Calculate simple metric
            contains = reference.lower() in answer.lower() or answer.lower() in reference.lower()
            
            # Store result
            result = {
                "question": question,
                "reference": reference,
                "generated": answer,
                "contains_answer": contains,
                "tau_q": debug_info["tau_q"],
                "filtered_docs_count": len(debug_info["filtered_docs"])
            }
            results.append(result)
            
            print(f"Answer: {answer}")
            print(f"Contains reference: {contains}")
            print(f"τq: {debug_info['tau_q']:.4f}")
            print(f"Filtered docs: {len(debug_info['filtered_docs'])}")
            
        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Error: {e}")
            results.append({
                "question": question,
                "reference": reference,
                "error": str(e)
            })
    
    # Calculate overall metrics
    metrics = {
        "total_questions": len(results),
        "success_count": sum(1 for r in results if r.get("contains_answer", False)),
        "error_count": sum(1 for r in results if "error" in r)
    }
    metrics["success_rate"] = metrics["success_count"] / metrics["total_questions"] if metrics["total_questions"] > 0 else 0
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": metrics,
                "results": results
            }, f, indent=2)
        print(f"Results saved to {output_file}")
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Success rate: {metrics['success_rate']:.2%} ({metrics['success_count']}/{metrics['total_questions']})")
    print(f"Error rate: {metrics['error_count'] / metrics['total_questions']:.2%}")
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DataMorgana questions using local Falcon with Wikipedia retriever")
    parser.add_argument(
        "--datamorgana_file", 
        required=True,
        help="Path to DataMorgana JSONL file"
    )
    parser.add_argument(
        "--falcon_model", 
        default="tiiuae/falcon-3-10B-instruct",
        help="Name or path of the Falcon model"
    )
    parser.add_argument(
        "--n", 
        type=float, 
        default=0.5,
        help="Adaptive judge bar parameter"
    )
    parser.add_argument(
        "--output", 
        help="Path to save evaluation results"
    )
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    else:
        # Default output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        args.output = f"results/datamorgana_eval_{timestamp}.json"
    
    print("Evaluating DataMorgana questions using local Falcon with Wikipedia retriever...")
    evaluate_datamorgana_with_local_falcon(
        datamorgana_file=args.datamorgana_file,
        falcon_model=args.falcon_model,
        n_value=args.n,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
