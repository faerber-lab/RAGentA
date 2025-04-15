import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm

# Import your existing classes
from main_rag import MAIN_RAG
from wikipedia_retriever import WikipediaRetriever
# The FalconAgent should be imported automatically by MAIN_RAG when needed


def evaluate_datamorgana_with_wikipedia(falcon_api_key, datamorgana_file, n_value=0.5, output_file=None):
    """
    Evaluate DataMorgana questions using Falcon model with Wikipedia retriever.
    
    This function:
    1. Loads DataMorgana QA pairs
    2. Initializes your existing Wikipedia retriever
    3. Sets up MAIN-RAG with Wikipedia retriever but Falcon for agents
    4. Processes each question and evaluates the answers
    
    Args:
        falcon_api_key: API key for Falcon model
        datamorgana_file: Path to DataMorgana QA pairs JSON file
        n_value: Adaptive judge bar adjustment parameter
        output_file: Path to save results
        
    Returns:
        Tuple of (metrics, detailed_results)
    """
    print(f"Loading DataMorgana questions from {qa_file}...")
    
    # Load JSONL file
    qa_pairs = load_jsonl(qa_file)
    print(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Initialize Wikipedia retriever
    print("Initializing Wikipedia retriever...")
    from wikipedia_retriever import WikipediaRetriever
    retriever = WikipediaRetriever()
    
    # Initialize MAIN-RAG with Falcon
    print(f"Initializing MAIN-RAG with Falcon (n={n_value})...")
    from main_rag import MAIN_RAG
    from falcon_agent import FalconAgent
    
    # Create Falcon agent
    falcon_agent = FalconAgent(falcon_api_key)
    
    # Create MAIN-RAG instance
    rag_system = MAIN_RAG(
        retriever=retriever,
        agent_model=falcon_agent,  # Pass the pre-initialized agent
        n=n_value
    )
    
    # Process QA pairs
    results = []
    for i, qa_pair in enumerate(tqdm(qa_pairs, desc="Evaluating DataMorgana questions")):
        question = qa_pair["question"]
        reference_answer = qa_pair["answer"]
        
        print(f"\n[{i+1}/{len(qa_pairs)}] Processing: {question}")
        
        try:
            # Get answer using MAIN-RAG
            answer, debug_info = rag_system.answer_query(question)
            
            # Calculate simple metrics
            contains = reference_answer.lower() in answer.lower() or answer.lower() in reference_answer.lower()
            
            # Save result
            result = {
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": answer,
                "contains_answer": contains,
                "tau_q": debug_info["tau_q"],
                "adjusted_tau_q": debug_info["adjusted_tau_q"],
                "filtered_docs_count": len(debug_info["filtered_docs"]),
                "total_docs_count": len(debug_info["scores"])
            }
            
            results.append(result)
            
            # Print interim results
            print(f"Generated answer: {answer}")
            print(f"Contains reference: {contains}")
            print(f"τq: {debug_info['tau_q']:.4f}, adjusted: {debug_info['adjusted_tau_q']:.4f}")
            print(f"Filtered docs: {len(debug_info['filtered_docs'])}/{len(debug_info['scores'])}")
            
        except Exception as e:
            print(f"Error processing question: {e}")
            results.append({
                "question": question,
                "reference_answer": reference_answer,
                "error": str(e)
            })
    
    # Calculate overall metrics
    metrics = {
        "total_questions": len(results),
        "success_count": sum(1 for r in results if r.get("contains_answer", False)),
        "error_count": sum(1 for r in results if "error" in r)
    }
    
    if metrics["total_questions"] > 0:
        metrics["success_rate"] = metrics["success_count"] / metrics["total_questions"]
    else:
        metrics["success_rate"] = 0.0
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "metrics": metrics,
                "results": results
            }, f, indent=2)
        print(f"Results saved to {output_file}")
    
    # Print overall metrics
    print("\n=== Overall Results ===")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Success rate: {metrics['success_rate']:.2%} ({metrics['success_count']}/{metrics['total_questions']})")
    print(f"Error count: {metrics['error_count']}")
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DataMorgana with Falcon and Wikipedia")
    parser.add_argument(
        "--falcon_key", required=True,
        help="API key for Falcon model"
    )
    parser.add_argument(
        "--qa_file", required=True,
        help="Path to DataMorgana QA pairs file"
    )
    parser.add_argument(
        "--n", type=float, default=0.5,
        help="Adaptive judge bar parameter"
    )
    parser.add_argument(
        "--output", 
        help="Path to save results (default: auto-generated with timestamp)"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Generate output filename with timestamp if not provided
    output_file = args.output
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/datamorgana_wikipedia_falcon_{timestamp}.json"
    
    # Run evaluation
    evaluate_datamorgana_with_wikipedia(
        falcon_api_key=args.falcon_key,
        datamorgana_file=args.qa_file,
        n_value=args.n,
        output_file=output_file
    )


if __name__ == "__main__":
    main()
