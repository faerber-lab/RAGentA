import argparse
import json
import time
import datetime
from tqdm import tqdm
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("main_rag_runner.log"), logging.StreamHandler()],
)
logger = logging.getLogger("MAIN_RAG_Runner")

# Import our components
from main_rag import MAIN_RAG
from hybrid_retriever import HybridRetriever


def load_datamorgana_questions(file_path="datamorgana_questions.jsonl"):
    """Load DataMorgana questions from JSONL file."""
    questions = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line)
                        questions.append(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON line: {e}")
        logger.info(f"Loaded {len(questions)} questions from {file_path}")
        return questions
    except FileNotFoundError:
        logger.error(f"DataMorgana questions file not found at {file_path}")
        return []


def format_output_for_submission(result):
    """Format result for formal submission requirements."""
    # Format as per specified requirements
    output = {
        "question_id": result.get("id", "unknown"),
        "question": result["question"],
        "answer": result["model_answer"],
        "supporting_passages": [
            {"text": doc_text, "doc_id": doc_id}
            for doc_text, doc_id in result["supporting_passages"]
        ],
        "full_prompt": result.get("agent3_prompt", ""),
    }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced MAIN-RAG with Hybrid Retrieval"
    )
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
    parser.add_argument(
        "--single_question",
        type=str,
        default=None,
        help="Process a single question instead of the entire dataset",
    )
    parser.add_argument(
        "--output_format",
        choices=["debug", "formal"],
        default="debug",
        help="Output format: 'debug' for detailed debug info or 'formal' for submission format",
    )
    args = parser.parse_args()

    # Initialize the hybrid retriever
    logger.info(f"Initializing hybrid retriever with alpha={args.alpha}...")
    retriever = HybridRetriever(alpha=args.alpha, top_k=args.top_k)

    # Initialize MAIN-RAG
    logger.info(f"Initializing enhanced MAIN-RAG with n={args.n}...")
    main_rag = MAIN_RAG(retriever, agent_model=args.model, n=args.n)

    # Create results directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/debug", exist_ok=True)
    os.makedirs("results/formal", exist_ok=True)

    # Process a single question if specified
    if args.single_question:
        logger.info(f"\nProcessing single question: {args.single_question}")
        start_time = time.time()

        try:
            # Process the query
            answer, debug_info = main_rag.answer_query(args.single_question)

            # Calculate processing time
            process_time = time.time() - start_time

            # Create result object
            result = {
                "id": "single_question",
                "question": args.single_question,
                "model_answer": answer,
                "tau_q": debug_info["tau_q"],
                "adjusted_tau_q": debug_info["adjusted_tau_q"],
                "filtered_count": len(debug_info["filtered_docs"]),
                "process_time": process_time,
                "completely_answered": debug_info["completely_answered"],
                "judge_response": debug_info["judge_response"],
                "follow_up_answers": debug_info["follow_up_answers"],
                "answer_with_citations": debug_info["answer_with_citations"],
                "supporting_passages": debug_info["supporting_passages"],
                "agent3_prompt": debug_info["agent3_prompt"],
            }

            logger.info(f"Answer: {answer}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(
                f"Tau_q: {debug_info['tau_q']:.4f} (adjusted: {debug_info['adjusted_tau_q']:.4f})"
            )
            logger.info(
                f"Filtered docs: {len(debug_info['filtered_docs'])}/{args.top_k}"
            )
            logger.info(f"Completely answered: {debug_info['completely_answered']}")

            # Save result based on format
            if args.output_format == "debug":
                output_file = f"results/debug/single_question_debug.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
            else:  # formal
                output_file = f"results/formal/single_question_formal.json"
                formal_output = format_output_for_submission(result)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(formal_output, f, indent=2)

            logger.info(f"Result saved to {output_file}")

        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)

        return

    # Load DataMorgana questions
    questions = load_datamorgana_questions(args.data_file)
    if not questions:
        logger.error("No questions found. Exiting.")
        return

    # Process each question
    debug_results = []
    formal_results = []

    for i, item in enumerate(questions):
        question_id = item.get("id", str(i + 1))
        logger.info(f"\nProcessing question {i+1}/{len(questions)}: {item['question']}")
        start_time = time.time()

        try:
            # Process the query
            answer, debug_info = main_rag.answer_query(item["question"])

            # Calculate processing time
            process_time = time.time() - start_time

            # Save result
            result = {
                "id": question_id,
                "question": item["question"],
                "reference_answer": item.get("answer", ""),
                "model_answer": answer,
                "tau_q": debug_info["tau_q"],
                "adjusted_tau_q": debug_info["adjusted_tau_q"],
                "filtered_count": len(debug_info["filtered_docs"]),
                "process_time": process_time,
                "completely_answered": debug_info["completely_answered"],
                "judge_response": debug_info["judge_response"],
                "follow_up_answers": debug_info["follow_up_answers"],
                "answer_with_citations": debug_info["answer_with_citations"],
                "supporting_passages": debug_info["supporting_passages"],
                "agent3_prompt": debug_info["agent3_prompt"],
            }
            debug_results.append(result)

            # Format for formal submission
            formal_result = format_output_for_submission(result)
            formal_results.append(formal_result)

            logger.info(f"Answer: {answer}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            logger.info(
                f"Tau_q: {debug_info['tau_q']:.4f} (adjusted: {debug_info['adjusted_tau_q']:.4f})"
            )
            logger.info(
                f"Filtered docs: {len(debug_info['filtered_docs'])}/{args.top_k}"
            )
            logger.info(f"Completely answered: {debug_info['completely_answered']}")

        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save results
    if debug_results:
        debug_output_file = (
            f"results/debug/enhanced_main_rag_n{args.n}_a{args.alpha}_{timestamp}.json"
        )
        with open(debug_output_file, "w", encoding="utf-8") as f:
            json.dump(debug_results, f, indent=2)
        logger.info(f"Debug results saved to {debug_output_file}")

    if formal_results:
        formal_output_file = (
            f"results/formal/enhanced_main_rag_n{args.n}_a{args.alpha}_{timestamp}.json"
        )
        with open(formal_output_file, "w", encoding="utf-8") as f:
            json.dump(formal_results, f, indent=2)
        logger.info(f"Formal results saved to {formal_output_file}")

    logger.info(f"\nProcessed {len(debug_results)} questions.")

    # Print summary statistics
    if debug_results:
        avg_time = sum(r["process_time"] for r in debug_results) / len(debug_results)
        avg_filtered = sum(r["filtered_count"] for r in debug_results) / len(
            debug_results
        )
        logger.info(f"Average processing time: {avg_time:.2f} seconds")
        logger.info(f"Average filtered documents: {avg_filtered:.1f}")


if __name__ == "__main__":
    main()
