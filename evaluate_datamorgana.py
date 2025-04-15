# evaluate_datamorgana.py

import json
import argparse
from tqdm import tqdm

def evaluate_with_datamorgana(rag_system, qa_file, output_file=None):
    """
    Evaluate MAIN-RAG using DataMorgana-generated QA pairs.
    
    Args:
        rag_system: Initialized MAIN_RAG instance
        qa_file: Path to DataMorgana QA pairs JSON file
        output_file: Optional path to save evaluation results
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Load QA pairs
    with open(qa_file, 'r') as f:
        qa_pairs = json.load(f)
    
    # Process each question
    results = []
    for qa_pair in tqdm(qa_pairs, desc="Evaluating"):
        question = qa_pair["question"]
        reference = qa_pair["answer"]
        
        try:
            # Generate answer using MAIN-RAG
            answer, debug_info = rag_system.answer_query(question)
            
            # Calculate basic metrics
            exact_match = answer.lower() == reference.lower()
            contains = reference.lower() in answer.lower() or answer.lower() in reference.lower()
            
            # Store result
            result = {
                "question": question,
                "reference": reference,
                "answer": answer,
                "exact_match": exact_match,
                "contains": contains,
                "tau_q": debug_info["tau_q"],
                "filtered_docs": len(debug_info["filtered_docs"])
            }
            results.append(result)
            
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
        "total": len(results),
        "exact_match": sum(r.get("exact_match", False) for r in results) / len(results),
        "contains": sum(r.get("contains", False) for r in results) / len(results),
        "error_rate": sum(1 for r in results if "error" in r) / len(results)
    }
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "metrics": metrics,
                "results": results
            }, f, indent=2)
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description="Evaluate MAIN-RAG with DataMorgana QA pairs")
    parser.add_argument("--qa_file", required=True, help="Path to DataMorgana QA pairs")
    parser.add_argument("--pinecone_key", required=True, help="Pinecone API key")
    parser.add_argument("--falcon_key", required=True, help="AI71 API key for Falcon")
    parser.add_argument("--n", type=float, default=0.5, help="Adaptive judge bar parameter")
    parser.add_argument("--output", help="Path to save evaluation results")
    args = parser.parse_args()
    
    # Initialize retriever
    from pinecone_retriever import PineconeRetriever
    retriever = PineconeRetriever(api_key=args.pinecone_key)
    
    # Initialize MAIN-RAG
    from main_rag import MAIN_RAG
    rag_system = MAIN_RAG(
        retriever=retriever,
        agent_model="falcon-3-10b-instruct",
        n=args.n,
        falcon_api_key=args.falcon_key
    )
    
    # Run evaluation
    metrics, results = evaluate_with_datamorgana(
        rag_system=rag_system,
        qa_file=args.qa_file,
        output_file=args.output
    )
    
    # Print summary
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
