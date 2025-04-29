import argparse
import os
import json
from datetime import datetime

from main_rag import MAIN_RAG
from pinecone_retriever import PineconeRetriever
from falcon_agent import FalconAgent


def evaluate_with_datamorgana(rag_system, qa_pairs, output_file=None):
    """
    Evaluate MAIN-RAG using DataMorgana-generated QA pairs.

    Args:
        rag_system: Initialized MAIN_RAG instance
        qa_pairs: List of QA pairs from DataMorgana
        output_file: Optional path to save evaluation results

    Returns:
        Dictionary of evaluation metrics
    """
    from tqdm import tqdm

    results = []
    for qa_pair in tqdm(qa_pairs, desc="Evaluating QA pairs"):
        question = qa_pair["question"]
        reference = qa_pair["answer"]

        try:
            # Generate answer using MAIN-RAG
            answer, debug_info = rag_system.answer_query(question)

            # Calculate simple metrics
            contains = (
                reference.lower() in answer.lower()
                or answer.lower() in reference.lower()
            )

            # Store result
            result = {
                "question": question,
                "reference": reference,
                "answer": answer,
                "contains_answer": contains,
                "tau_q": debug_info["tau_q"],
                "filtered_docs_count": len(debug_info["filtered_docs"]),
            }
            results.append(result)

            print(f"\nQuestion: {question}")
            print(f"Answer: {answer}")
            print(f"Contains reference: {contains}")
            print(
                f"τq: {debug_info['tau_q']:.4f}, Filtered docs: {len(debug_info['filtered_docs'])}"
            )

        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Error: {e}")
            results.append(
                {"question": question, "reference": reference, "error": str(e)}
            )

    # Calculate overall metrics
    metrics = {
        "total_questions": len(results),
        "success_count": sum(1 for r in results if r.get("contains_answer", False)),
        "error_count": sum(1 for r in results if "error" in r),
    }
    metrics["success_rate"] = (
        metrics["success_count"] / metrics["total_questions"]
        if metrics["total_questions"] > 0
        else 0
    )

    # Save results if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)

    return metrics, results


def main():
    parser = argparse.ArgumentParser(
        description="Run MAIN-RAG with Pinecone and Falcon"
    )
    parser.add_argument("--pinecone_key", required=True, help="Pinecone API key")
    parser.add_argument("--falcon_key", required=True, help="AI71 API key for Falcon")
    parser.add_argument(
        "--n", type=float, default=0.5, help="Adaptive judge bar parameter"
    )
    parser.add_argument(
        "--qa_file",
        help="Path to DataMorgana QA pairs JSON file for evaluation (optional)",
    )
    parser.add_argument(
        "--query", help="Single query to answer (if not using --qa_file)"
    )
    args = parser.parse_args()

    # Initialize the Pinecone retriever
    print("Initializing Pinecone retriever...")
    retriever = PineconeRetriever(
        api_key=args.pinecone_key,
        index_name="fineweb10bt-512-0w-e5-base-v2",
        namespace="default",
    )

    # Initialize MAIN-RAG
    print(f"Initializing MAIN-RAG with n={args.n}...")
    rag_system = MAIN_RAG(
        retriever=retriever,
        agent_model="falcon-3-10b-instruct",
        n=args.n,
        falcon_api_key=args.falcon_key,
    )

    # Create results directory
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run evaluation or answer a single query
    if args.qa_file:
        print(f"Loading QA pairs from {args.qa_file}...")
        with open(args.qa_file, "r") as f:
            qa_pairs = json.load(f)

        output_file = f"results/evaluation_{timestamp}.json"
        print(f"Evaluating {len(qa_pairs)} QA pairs...")

        metrics, results = evaluate_with_datamorgana(
            rag_system=rag_system, qa_pairs=qa_pairs, output_file=output_file
        )

        print("\nEvaluation Results:")
        print(
            f"Success rate: {metrics['success_rate']:.2%} ({metrics['success_count']}/{metrics['total_questions']})"
        )
        print(f"Results saved to {output_file}")

    elif args.query:
        print(f"Processing query: {args.query}")
        answer, debug_info = rag_system.answer_query(args.query)

        print("\nResults:")
        print(f"Query: {args.query}")
        print(f"Answer: {answer}")
        print(f"τq: {debug_info['tau_q']:.4f}")
        print(f"Filtered documents: {len(debug_info['filtered_docs'])}")

        # Save results
        output_file = f"results/query_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "query": args.query,
                    "answer": answer,
                    "tau_q": debug_info["tau_q"],
                    "filtered_docs_count": len(debug_info["filtered_docs"]),
                },
                f,
                indent=2,
            )

        print(f"Results saved to {output_file}")

    else:
        print("Error: Either --qa_file or --query must be specified")


if __name__ == "__main__":
    main()
