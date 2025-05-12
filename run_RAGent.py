import argparse
import json
import time
import datetime
import os
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ragent_runner.log"), logging.StreamHandler()],
)
logger = logging.getLogger("RAGENT_Runner")

# Import our components
from RAGent import RAGENT
from hybrid_retriever import HybridRetriever


def load_datamorgana_questions(file_path):
    """
    Load DataMorgana questions from either JSON or JSONL format.

    Args:
        file_path: Path to the JSON or JSONL file

    Returns:
        List of question dictionaries
    """
    # Determine if file is likely JSON or JSONL based on extension
    is_jsonl = file_path.lower().endswith(".jsonl")

    try:
        questions = []

        # JSONL format: each line is a separate JSON object
        if is_jsonl:
            logger.info(f"Loading questions from JSONL file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    try:
                        question = json.loads(line)

                        # Add line number as ID if not present
                        if "id" not in question:
                            question["id"] = line_num

                        questions.append(question)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON at line {line_num}: {e}")

        # JSON format: entire file is a single JSON object or array
        else:
            logger.info(f"Loading questions from JSON file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # If the JSON is an array, use it directly
                if isinstance(data, list):
                    questions = data

                    # Add indices as IDs if not present
                    for i, question in enumerate(questions):
                        if "id" not in question:
                            question["id"] = i + 1
                # If the JSON is an object, look for a questions field or use as a single question
                elif isinstance(data, dict):
                    if "questions" in data:
                        questions = data["questions"]
                    elif "question" in data:
                        questions = [data]
                    else:
                        # Treat the entire object as a single question
                        questions = [data]

        logger.info(f"Loaded {len(questions)} questions")
        return questions

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading questions: {e}")
        return []


def format_result_to_schema(result):
    """
    Format a result dictionary to match the required Answer.json schema.

    Args:
        result: Dictionary containing the query results

    Returns:
        Dictionary formatted according to the schema
    """
    # Extract the passages and document IDs
    supporting_passages = result.get("supporting_passages", [])

    # Format passages according to schema
    formatted_passages = []
    doc_passage_map = {}

    for passage_tuple in supporting_passages:
        # Each passage is a tuple of (text, doc_id)
        passage_text, doc_id = passage_tuple

        # Group passages by text to combine multiple doc_IDs
        if passage_text not in doc_passage_map:
            doc_passage_map[passage_text] = []

        if doc_id not in doc_passage_map[passage_text]:
            doc_passage_map[passage_text].append(doc_id)

    # Convert the map to the required format
    for passage_text, doc_ids in doc_passage_map.items():
        formatted_passages.append({"passage": passage_text, "doc_IDs": doc_ids})

    # Create the formatted result dictionary
    formatted_result = {
        "id": result.get("id", 0),  # Default to 0 if no ID
        "question": result.get("question", ""),
        "passages": formatted_passages,
        "final_prompt": result.get("agent3_prompt", ""),
        "answer": result.get("model_answer", ""),
    }

    return formatted_result


def write_results_to_jsonl(results, output_file):
    """
    Write a list of results to a JSONL file.

    Args:
        results: List of result dictionaries
        output_file: Path to output file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            # Format each result according to schema
            formatted_result = format_result_to_schema(result)

            # Write as a single JSON line
            f.write(json.dumps(formatted_result, ensure_ascii=False) + "\n")

    logger.info(f"Results written to {output_file}")


def write_result_to_json(result, output_file):
    """
    Write a single result to a JSON file.

    Args:
        result: Result dictionary
        output_file: Path to output file
    """
    # Format result according to schema
    formatted_result = format_result_to_schema(result)

    # Write as a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)

    logger.info(f"Result written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced RAGent with Hybrid Retrieval"
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
        default="datamorgana_questions.jsonl",
        help="File containing DataMorgana questions (JSON or JSONL)",
    )
    parser.add_argument(
        "--single_question",
        type=str,
        default=None,
        help="Process a single question instead of the entire dataset",
    )
    parser.add_argument(
        "--output_format",
        choices=["json", "jsonl", "debug"],
        default="jsonl",
        help="Output format: 'json' for single file, 'jsonl' for line-delimited JSON, or 'debug' for detailed output",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    args = parser.parse_args()

    # Initialize the hybrid retriever
    logger.info(f"Initializing hybrid retriever with alpha={args.alpha}...")
    retriever = HybridRetriever(alpha=args.alpha, top_k=args.top_k)

    # Initialize RAGent
    logger.info(f"Initializing enhanced RAGent with n={args.n}...")
    ragent = RAGENT(retriever, agent_model=args.model, n=args.n)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "debug"), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process a single question if specified
    if args.single_question:
        logger.info(f"\nProcessing single question: {args.single_question}")
        start_time = time.time()

        try:
            # Process the query
            answer, debug_info = ragent.answer_query(args.single_question)

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
                "completely_answered": debug_info.get("completely_answered", False),
                "supporting_passages": debug_info["supporting_passages"],
                "agent3_prompt": debug_info["agent3_prompt"],
                # Add claim_analysis instead of judge_response if it exists
                "claim_analysis": debug_info.get("claim_analysis", ""),
                "question_aspects": debug_info.get("question_aspects", []),
                "follow_up_questions": debug_info.get("follow_up_questions", []),
            }

            logger.info(f"Answer: {answer}")
            logger.info(f"Processing time: {process_time:.2f} seconds")

            # Save result based on format
            if args.output_format == "debug":
                debug_output_file = os.path.join(
                    args.output_dir, "debug", f"single_question_debug_{timestamp}.json"
                )
                with open(debug_output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Debug result saved to {debug_output_file}")
            else:  # json
                output_file = os.path.join(
                    args.output_dir, f"single_question_{timestamp}.json"
                )
                write_result_to_json(result, output_file)

        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)

        return

    # Load DataMorgana questions
    questions = load_datamorgana_questions(args.data_file)
    if not questions:
        logger.error("No questions found. Exiting.")
        return

    # Process each question
    results = []

    for i, item in enumerate(questions):
        question_id = item.get("id", i + 1)
        logger.info(f"\nProcessing question {i+1}/{len(questions)}: {item['question']}")
        start_time = time.time()

        try:
            # Process the query
            answer, debug_info = ragent.answer_query(item["question"])

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
                "completely_answered": debug_info.get("completely_answered", False),
                "supporting_passages": debug_info["supporting_passages"],
                "agent3_prompt": debug_info["agent3_prompt"],
                # Add claim_analysis instead of judge_response if it exists
                "claim_analysis": debug_info.get("claim_analysis", ""),
                "question_aspects": debug_info.get("question_aspects", []),
                "follow_up_questions": debug_info.get("follow_up_questions", []),
            }
            results.append(result)

            logger.info(f"Answer: {answer}")
            logger.info(f"Processing time: {process_time:.2f} seconds")

            # Save debug information
            debug_output_file = os.path.join(
                args.output_dir,
                "debug",
                f"question_{question_id}_debug_{timestamp}.json",
            )
            with open(debug_output_file, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}", exc_info=True)

    # Save all results
    if results:
        if args.output_format == "jsonl":
            output_file = os.path.join(args.output_dir, f"answers_{timestamp}.jsonl")
            write_results_to_jsonl(results, output_file)
        elif args.output_format == "json":
            # Save each result as a separate JSON file
            for result in results:
                question_id = result["id"]
                output_file = os.path.join(
                    args.output_dir, f"answer_{question_id}_{timestamp}.json"
                )
                write_result_to_json(result, output_file)
        else:  # debug
            output_file = os.path.join(
                args.output_dir, "debug", f"all_results_debug_{timestamp}.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Debug results saved to {output_file}")

    logger.info(f"\nProcessed {len(results)} questions.")

    # Print summary statistics
    if results:
        avg_time = sum(r["process_time"] for r in results) / len(results)
        avg_filtered = sum(r["filtered_count"] for r in results) / len(results)
        logger.info(f"Average processing time: {avg_time:.2f} seconds")
        logger.info(f"Average filtered documents: {avg_filtered:.1f}")


if __name__ == "__main__":
    main()
