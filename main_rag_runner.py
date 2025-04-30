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
import datetime
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
        # Try loading as JSONL if JSON fails
        try:
            data = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            if data:
                return data
        except:
            pass

        print(f"Error: DataMorgana questions file not found at {file_path}")
        return []


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
        default="datamorgana_questions.jsonl",
        help="File containing DataMorgana questions",
    )
    parser.add_argument(
        "--use_citations",
        action="store_true",
        help="Use the citation-based answer generation with faithfulness checking",
    )
    parser.add_argument(
        "--skip_faithfulness",
        action="store_true",
        help="Skip faithfulness checking when using citations",
    )
    parser.add_argument(
        "--output_prefix", type=str, default="", help="Prefix for the output file name"
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
            # Process the query using either the original method or the enhanced citation method
            if args.use_citations:
                answer, debug_info = main_rag.answer_query_with_citations(
                    item["question"],
                    choices=None,
                    use_faithfulness_judge=not args.skip_faithfulness,
                )
            else:
                answer, debug_info = main_rag.answer_query(item["question"])

            # Calculate processing time
            process_time = time.time() - start_time

            # Save result
            result = {
                "question": item["question"],
                "reference_answer": item.get("answer", ""),
                "model_answer": answer,
                "tau_q": debug_info.get("tau_q", 0),
                "adjusted_tau_q": debug_info.get("adjusted_tau_q", 0),
                "filtered_count": len(debug_info.get("filtered_docs", [])),
                "process_time": process_time,
            }

            # Add citation info if available
            if "faithfulness_results" in debug_info:
                result["faithfulness"] = {
                    "action": debug_info["faithfulness_results"]["action"],
                    "valid_count": len(
                        debug_info["faithfulness_results"]["valid_citations"]
                    ),
                    "invalid_count": len(
                        debug_info["faithfulness_results"]["invalid_citations"]
                    ),
                    "hallucination_count": len(
                        debug_info["faithfulness_results"]["hallucinations"]
                    ),
                    "feedback": debug_info["faithfulness_results"]["feedback"],
                }

            results.append(result)

            print(f"Answer: {answer}")
            print(f"Processing time: {process_time:.2f} seconds")
            print(
                f"Tau_q: {debug_info.get('tau_q', 0):.4f} (adjusted: {debug_info.get('adjusted_tau_q', 0):.4f})"
            )
            print(
                f"Filtered docs: {len(debug_info.get('filtered_docs', []))}/{args.top_k}"
            )

            # Print faithfulness info if available
            if "faithfulness_results" in debug_info:
                faith_results = debug_info["faithfulness_results"]
                print(f"Faithfulness action: {faith_results['action']}")
                print(f"Valid citations: {len(faith_results['valid_citations'])}")
                print(f"Invalid citations: {len(faith_results['invalid_citations'])}")
                print(f"Hallucinations: {len(faith_results['hallucinations'])}")

        except Exception as e:
            print(f"Error processing question: {e}")
            import traceback

            traceback.print_exc()

    # Generate timestamp for the output file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output filename with prefix and indicators
    citation_indicator = "with_citations" if args.use_citations else "standard"
    if args.use_citations and args.skip_faithfulness:
        citation_indicator += "_no_faithfulness"

    output_file = f"results/{args.output_prefix}main_rag_{citation_indicator}_n{args.n}_a{args.alpha}_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessed {len(results)} questions. Results saved to {output_file}")

    # Print summary statistics
    if results:
        avg_time = sum(r["process_time"] for r in results) / len(results)
        avg_filtered = sum(r["filtered_count"] for r in results) / len(results)
        print(f"Average processing time: {avg_time:.2f} seconds")
        print(f"Average filtered documents: {avg_filtered:.1f}")

        # Print faithfulness statistics if available
        faith_results = [r for r in results if "faithfulness" in r]
        if faith_results:
            avg_valid = sum(
                r["faithfulness"]["valid_count"] for r in faith_results
            ) / len(faith_results)
            avg_invalid = sum(
                r["faithfulness"]["invalid_count"] for r in faith_results
            ) / len(faith_results)
            avg_hallucinations = sum(
                r["faithfulness"]["hallucination_count"] for r in faith_results
            ) / len(faith_results)

            print(f"Average valid citations: {avg_valid:.1f}")
            print(f"Average invalid citations: {avg_invalid:.1f}")
            print(f"Average hallucinations: {avg_hallucinations:.1f}")

            # Count faithfulness actions
            actions = {}
            for r in faith_results:
                action = r["faithfulness"]["action"]
                actions[action] = actions.get(action, 0) + 1

            print("Faithfulness actions:")
            for action, count in actions.items():
                print(f"  {action}: {count} ({count/len(faith_results)*100:.1f}%)")


if __name__ == "__main__":
    main()
