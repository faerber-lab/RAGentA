# main_rag_runner.py
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
import json
import time
from tqdm import tqdm

# Import our components
from main_rag import MAIN_RAG
from hybrid_retriever import HybridRetriever


def load_datamorgana_questions(file_path="datamorgana_questions.json"):
    """Load DataMorgana questions."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: DataMorgana questions file not found at {file_path}")
        return []


# Only the modified part is shown - this should be integrated with your existing code
def main():
    parser = argparse.ArgumentParser(description="MAIN-RAG with Hybrid Retrieval")
    parser.add_argument(
        "--model",
        type=str,
        default="tiiuae/falcon-3-10b-instruct",
        help="Model for LLM agents",
    )
    parser.add_argument(
        "--n", type=float, default=0.5, help="Adjustment factor for adaptive judge bar"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="Weight for semantic search (0-1)"
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="datamorgana_questions.json",
        help="File containing DataMorgana questions",
    )
    args = parser.parse_args()

    # Initialize the hybrid retriever
    print(f"Initializing hybrid retriever with alpha={args.alpha}...")
    retriever = HybridRetriever(alpha=args.alpha, top_k=args.top_k)

    # Initialize MAIN-RAG
    print(f"Initializing MAIN-RAG with n={args.n}...")
    main_rag = MAIN_RAG(retriever, agent_model=args.model, n=args.n)

    # Load DataMorgana questions
    questions = load_datamorgana_questions(args.data_file)
    if not questions:
        print("No questions found. Exiting.")
        return

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Process each question
    results = []
    for i, item in enumerate(questions):
        print(f"\nProcessing question {i+1}/{len(questions)}: {item['question']}")
        start_time = time.time()

        try:
            # Process the query
            answer, debug_info = main_rag.answer_query(item["question"])

            # Calculate processing time
            process_time = time.time() - start_time

            # Save result
            result = {
                "question": item["question"],
                "reference_answer": item["answer"],
                "model_answer": answer,
                "tau_q": debug_info["tau_q"],
                "adjusted_tau_q": debug_info["adjusted_tau_q"],
                "filtered_count": len(debug_info["filtered_docs"]),
                "process_time": process_time,
                # Add filtered documents with their content for evaluation
                "filtered_docs": [
                    (doc, float(score)) for doc, score in debug_info["filtered_docs"]
                ],
            }
            results.append(result)

            print(f"Answer: {answer}")
            print(f"Processing time: {process_time:.2f} seconds")
            print(
                f"Tau_q: {debug_info['tau_q']:.4f} (adjusted: {debug_info['adjusted_tau_q']:.4f})"
            )
            print(f"Filtered docs: {len(debug_info['filtered_docs'])}/{args.top_k}")

        except Exception as e:
            print(f"Error processing question: {e}")

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/main_rag_hybrid_n{args.n}_a{args.alpha}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessed {len(results)} questions. Results saved to {output_file}")

    # Print summary statistics
    if results:
        avg_time = sum(r["process_time"] for r in results) / len(results)
        avg_filtered = sum(r["filtered_count"] for r in results) / len(results)
        print(f"Average processing time: {avg_time:.2f} seconds")
        print(f"Average filtered documents: {avg_filtered:.1f}")


if __name__ == "__main__":
    main()
