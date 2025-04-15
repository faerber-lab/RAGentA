#!/usr/bin/env python
"""
MAIN-RAG Runner Script
This script provides a simple CLI for interactively testing the MAIN-RAG system.
"""

import argparse
import json
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="MAIN-RAG: Interactive testing")
    parser.add_argument(
        "--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Model to use for agents"
    )
    parser.add_argument(
        "--n", type=float, default=0.0, help="Adjustment factor for adaptive judge bar"
    )
    parser.add_argument(
        "--retriever_model", type=str, default="facebook/contriever-msmarco", 
        help="Model for retriever embeddings"
    )
    parser.add_argument(
        "--max_passages", type=int, default=100000, 
        help="Maximum number of passages to process in Wikipedia retriever"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--query", type=str, default=None, help="Query to process (for non-interactive mode)"
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save results to file"
    )
    args = parser.parse_args()

    # Initialize the retriever
    print("Initializing retriever...")
    from wikipedia_retriever import WikipediaRetriever
    retriever = WikipediaRetriever(
        model_name=args.retriever_model,
        max_passages=args.max_passages
    )

    # Initialize MAIN-RAG
    print(f"Initializing MAIN-RAG with model {args.model} and n={args.n}...")
    from main_rag import MAIN_RAG
    main_rag = MAIN_RAG(retriever, agent_model=args.model, n=args.n)

    if args.interactive:
        # Interactive mode
        print("\n===== MAIN-RAG Interactive Mode =====")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'adjust n <value>' to change the adaptive judge bar adjustment factor.")
        
        while True:
            query = input("\nEnter your query: ")
            
            if query.lower() in ["exit", "quit"]:
                break
                
            if query.lower().startswith("adjust n "):
                try:
                    n_value = float(query.split("adjust n ")[1])
                    main_rag.n = n_value
                    print(f"Adjusted n to {n_value}")
                except:
                    print("Invalid n value. Usage: adjust n 0.5")
                continue
            
            print("Processing query...")
            answer, debug_info = main_rag.answer_query(query)
            
            print("\n==== Results ====")
            print(f"Answer: {answer}")
            print(f"\nDebug Information:")
            print(f"- Adaptive Judge Bar (τq): {debug_info['tau_q']:.4f}")
            print(f"- Adjusted Judge Bar: {debug_info['adjusted_tau_q']:.4f}")
            print(f"- Standard Deviation: {debug_info['sigma']:.4f}")
            print(f"- Number of filtered documents: {len(debug_info['filtered_docs'])}")
            
            if args.save_results:
                save_results(query, answer, debug_info)
    
    elif args.query:
        # Single query mode
        print(f"Processing query: {args.query}")
        answer, debug_info = main_rag.answer_query(args.query)
        
        print("\n==== Results ====")
        print(f"Answer: {answer}")
        print(f"\nDebug Information:")
        print(f"- Adaptive Judge Bar (τq): {debug_info['tau_q']:.4f}")
        print(f"- Adjusted Judge Bar: {debug_info['adjusted_tau_q']:.4f}")
        print(f"- Standard Deviation: {debug_info['sigma']:.4f}")
        print(f"- Number of filtered documents: {len(debug_info['filtered_docs'])}")
        
        # Print top 3 filtered documents
        print("\nTop filtered documents:")
        for i, (doc, score) in enumerate(debug_info['filtered_docs'][:3]):
            print(f"\n{i+1}. [Score: {score:.4f}]")
            print(f"{doc[:300]}...")  # Print just the beginning
        
        if args.save_results:
            save_results(args.query, answer, debug_info)
    
    else:
        print("Error: Either --interactive or --query must be specified.")
        parser.print_help()


def save_results(query, answer, debug_info, output_dir="results"):
    """Save query results to a file for analysis."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{timestamp}.json"

    result = {
        "query": query,
        "answer": answer,
        "debug_info": {
            "tau_q": debug_info["tau_q"],
            "adjusted_tau_q": debug_info["adjusted_tau_q"],
            "sigma": debug_info["sigma"],
            "filtered_count": len(debug_info["filtered_docs"]),
            "filtered_docs": [(doc[:300], score) for doc, score in debug_info["filtered_docs"]],  # Truncate docs
        },
    }

    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {os.path.join(output_dir, filename)}")


if __name__ == "__main__":
    main()
