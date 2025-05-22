import argparse
import json
import time
from tqdm import tqdm
import logging
import os
import random
import string
import datetime
import re


# Generate a unique ID for log filename
def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/retrieval_eval_{timestamp}_{random_str}.log"


# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging with unique filename
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("RETRIEVAL_EVAL")

# Import your retriever and evaluation functions
from hybrid_retriever import HybridRetriever
from retriever_evaluation import (
    evaluate_corpus_rag_mrr,
    evaluate_corpus_rag_recall,
    evaluate_corpus_rag_precision,
    evaluate_corpus_rag_f1,
)


def load_questions(file_path):
    """Load questions from JSON or JSONL file."""
    logger.info(f"Loading questions from: {file_path}")

    # Determine file format
    is_jsonl = file_path.lower().endswith(".jsonl")

    try:
        questions = []
        if is_jsonl:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "question" in item:
                            questions.append(item)
        else:  # JSON format
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    questions = [q for q in data if "question" in q]
                elif isinstance(data, dict) and "questions" in data:
                    questions = [q for q in data["questions"] if "question" in q]
                elif "question" in data:
                    questions = [data]

        logger.info(f"Loaded {len(questions)} questions")
        return questions

    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return []


def load_golden_documents(file_path):
    """Load golden documents from JSON or JSONL file."""
    logger.info(f"Loading golden documents from: {file_path}")

    # Determine file format
    is_jsonl = file_path.lower().endswith(".jsonl")

    try:
        golden_data = {}
        if is_jsonl:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "question" in item and "document_ids" in item:
                            golden_data[item["question"]] = item["document_ids"]
        else:  # JSON format
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if "question" in item and "document_ids" in item:
                            golden_data[item["question"]] = item["document_ids"]
                elif isinstance(data, dict):
                    if "question" in data and "document_ids" in data:
                        golden_data[data["question"]] = data["document_ids"]

        logger.info(f"Loaded golden documents for {len(golden_data)} questions")
        return golden_data

    except Exception as e:
        logger.error(f"Error loading golden documents: {e}")
        return {}


def normalize_id(doc_id):
    """Extract the UUID part from document IDs with different formats."""
    # Pattern to match UUIDs inside document IDs
    uuid_pattern = r"<urn:uuid:([0-9a-f-]+)>"

    # Try to extract the UUID
    match = re.search(uuid_pattern, doc_id)
    if match:
        # Return just the UUID part
        return match.group(1)
    return doc_id  # Return original if no UUID found


def run_evaluation(retriever, questions, golden_docs, ks=[5, 10, 20, 50, 100]):
    """
    Run evaluation with ID normalization.
    """
    logger.info(f"Running evaluation on {len(questions)} questions")

    # Lists to store all results
    all_retrieved_ids = []
    all_golden_ids = []

    # Detailed comparison data for debugging
    comparison_data = []

    # Process each question
    for question in tqdm(questions, desc="Retrieving documents"):
        query = question["question"]

        # Skip if no golden documents
        if query not in golden_docs:
            logger.warning(f"No golden documents for query: {query}")
            continue

        try:
            # Retrieve documents
            retrieved_docs = retriever.retrieve(query, top_k=max(ks))

            # Extract document IDs and normalize them
            raw_retrieved_ids = [doc_id for _, doc_id in retrieved_docs]
            normalized_retrieved_ids = [
                normalize_id(doc_id) for doc_id in raw_retrieved_ids
            ]

            # Get golden document IDs and normalize them
            raw_golden_ids = golden_docs[query]
            normalized_golden_ids = [normalize_id(doc_id) for doc_id in raw_golden_ids]

            # Calculate intersection using normalized IDs
            intersection = set(normalized_retrieved_ids[: max(ks)]).intersection(
                set(normalized_golden_ids)
            )

            # Store detailed comparison for this query
            query_comparison = {
                "query": query,
                "raw_golden_ids": raw_golden_ids,
                "normalized_golden_ids": normalized_golden_ids,
                "raw_retrieved_ids": raw_retrieved_ids[: max(ks)],
                "normalized_retrieved_ids": normalized_retrieved_ids[: max(ks)],
                "intersection": list(intersection),
                "intersection_count": len(intersection),
                "golden_count": len(normalized_golden_ids),
                "retrieved_count": min(len(normalized_retrieved_ids), max(ks)),
            }
            comparison_data.append(query_comparison)

            # Log comparison details
            logger.info(f"Query: {query}")
            logger.info(f"  Normalized Golden IDs count: {len(normalized_golden_ids)}")
            logger.info(
                f"  Normalized Retrieved IDs count: {len(normalized_retrieved_ids[:max(ks)])}"
            )
            logger.info(f"  Intersection count: {len(intersection)}")

            # Log the first few IDs for comparison
            if raw_golden_ids:
                logger.info(
                    f"  First golden ID: {raw_golden_ids[0]} → {normalized_golden_ids[0]}"
                )
            if raw_retrieved_ids:
                logger.info(
                    f"  First retrieved ID: {raw_retrieved_ids[0]} → {normalized_retrieved_ids[0]}"
                )

            if intersection:
                logger.info(f"  Matched normalized IDs: {list(intersection)[:5]}")
            else:
                logger.info("  NO MATCHES FOUND")

            # For metrics calculation, use the normalized IDs
            all_retrieved_ids.append(normalized_retrieved_ids)
            all_golden_ids.append(normalized_golden_ids)

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")

    # Save detailed comparison data
    with open("retrieval_comparison_data.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Calculate metrics for different k values
    results = {}

    for k in ks:
        logger.info(f"Calculating metrics for k={k}")

        # Calculate standard metrics
        mrr = evaluate_corpus_rag_mrr(all_retrieved_ids, all_golden_ids, k=k)
        recall = evaluate_corpus_rag_recall(all_retrieved_ids, all_golden_ids, k=k)
        precision = evaluate_corpus_rag_precision(
            all_retrieved_ids, all_golden_ids, k=k
        )
        f1 = evaluate_corpus_rag_f1(all_retrieved_ids, all_golden_ids, k=k)

        # Calculate additional metrics for this k value
        queries_with_matches_k = 0
        total_matches_k = 0

        for retrieved_ids, golden_ids in zip(all_retrieved_ids, all_golden_ids):
            # Consider only the top-k retrieved documents
            intersection_k = set(retrieved_ids[:k]).intersection(set(golden_ids))
            if len(intersection_k) > 0:
                queries_with_matches_k += 1
            total_matches_k += len(intersection_k)

        # Calculate per-k additional metrics
        total_queries = len(all_retrieved_ids)
        queries_with_matches_ratio = (
            queries_with_matches_k / total_queries if total_queries > 0 else 0
        )
        average_matches_k = total_matches_k / total_queries if total_queries > 0 else 0

        # Store all metrics together for this k
        results[k] = {
            "MRR": mrr,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "queries_with_matches": queries_with_matches_k,
            "queries_with_matches_ratio": queries_with_matches_ratio,
            "average_matches": average_matches_k,
        }

        logger.info(
            f"k={k}: MRR={mrr:.4f}, Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}"
        )
        logger.info(
            f"k={k}: queries_with_matches={queries_with_matches_k}/{total_queries} ({queries_with_matches_ratio:.4f}), average_matches={average_matches_k:.4f}"
        )

    # Add overall comparison summary to results (using max k)
    results["comparison_summary"] = {
        "total_queries": len(comparison_data),
        "queries_with_matches": sum(
            1 for item in comparison_data if item["intersection_count"] > 0
        ),
        "average_golden_ids": (
            sum(item["golden_count"] for item in comparison_data) / len(comparison_data)
            if comparison_data
            else 0
        ),
        "average_matches": (
            sum(item["intersection_count"] for item in comparison_data)
            / len(comparison_data)
            if comparison_data
            else 0
        ),
    }

    return results, comparison_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate Retrieval System")
    parser.add_argument(
        "--question_file",
        type=str,
        required=True,
        help="Path to question file (JSON or JSONL)",
    )
    parser.add_argument(
        "--golden_file",
        type=str,
        required=True,
        help="Path to file with golden document IDs",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="Weight for semantic search (0-1)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="retrieval_eval_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="5,10,20,50,100",
        help="Comma-separated list of k values to evaluate",
    )

    args = parser.parse_args()

    # Load questions and golden documents
    questions = load_questions(args.question_file)
    golden_docs = load_golden_documents(args.golden_file)

    if not questions or not golden_docs:
        logger.error("Failed to load questions or golden documents. Exiting.")
        return

    # Parse k values
    ks = [int(k) for k in args.ks.split(",")]

    # Initialize retriever
    logger.info(f"Initializing hybrid retriever with alpha={args.alpha}")
    retriever = HybridRetriever(alpha=args.alpha)

    # Run evaluation with enhanced comparison data
    start_time = time.time()
    results, comparison_data = run_evaluation(retriever, questions, golden_docs, ks=ks)
    elapsed_time = time.time() - start_time

    # Add metadata to results
    output_results = {
        "metadata": {
            "alpha": args.alpha,
            "num_questions": len(questions),
            "evaluation_time": elapsed_time,
        },
        "metrics": results,
    }

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output_results, f, indent=2)

    # Save detailed comparison data
    comparison_file = os.path.join(
        os.path.dirname(args.output_file), f"comparison_data_{int(time.time())}.json"
    )
    with open(comparison_file, "w") as f:
        json.dump(comparison_data, f, indent=2)

    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to {args.output_file}")
    logger.info(f"Comparison data saved to {comparison_file}")

    # Print summary
    print("\nSummary of Results:")
    print("===================")
    for k, metrics in results.items():
        print(f"k={k}:")
        print(f"  MRR: {metrics['MRR']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  F1: {metrics['F1']:.4f}")
        print()


if __name__ == "__main__":
    main()
