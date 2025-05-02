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
    args = parser.parse_args()

    # Initialize the hybrid retriever
    print(f"Initializing hybrid retriever with alpha={args.alpha}...")
    retriever = HybridRetriever(alpha=args.alpha, top_k=args.top_k)

    # Initialize MAIN-RAG
    print(f"Initializing enhanced MAIN-RAG with n={args.n}...")
    main_rag = MAIN_RAG(retriever, agent_model=args.model, n=args.n)

    # Process a single question if specified
    if args.single_question:
        print(f"\nProcessing single question: {args.single_question}")
        start_time = time.time()

        try:
            # Process the query
            answer, debug_info = main_rag.answer_query(args.single_question)

            # Calculate processing time
            process_time = time.time() - start_time

            # Create result object
            result = {
                "question": args.single_question,
                "model_answer": answer,
                "tau_q": debug_info["tau_q"],
                "adjusted_tau_q": debug_info["adjusted_tau_q"],
                "filtered_count": len(debug_info["filtered_docs"]),
                "process_time": process_time,
                "completely_answered": debug_info["completely_answered"],
                "judge_response": debug_info["judge_response"],
                "follow_up_answers": debug_info["follow_up_answers"],
            }

            print(f"Answer: {answer}")
            print(f"Processing time: {process_time:.2f} seconds")
            print(
                f"Tau_q: {debug_info['tau_q']:.4f} (adjusted: {debug_info['adjusted_tau_q']:.4f})"
            )
            print(f"Filtered docs: {len(debug_info['filtered_docs'])}/{args.top_k}")
            print(f"Completely answered: {debug_info['completely_answered']}")

            # Save single result
            output_file = f"results/enhanced_main_rag_single_question.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"Result saved to {output_file}")

        except Exception as e:
            print(f"Error processing question: {e}")

        return

    # Load DataMorgana questions
    questions = load_datamorgana_questions(args.data_file)
    if not questions:
        print("No questions found. Exiting.")
        return

    # Create results directory
    import os

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
                "completely_answered": debug_info["completely_answered"],
                "judge_response": debug_info["judge_response"],
                "follow_up_answers": debug_info["follow_up_answers"],
            }
            results.append(result)

            print(f"Answer: {answer}")
            print(f"Processing time: {process_time:.2f} seconds")
            print(
                f"Tau_q: {debug_info['tau_q']:.4f} (adjusted: {debug_info['adjusted_tau_q']:.4f})"
            )
            print(f"Filtered docs: {len(debug_info['filtered_docs'])}/{args.top_k}")
            print(f"Completely answered: {debug_info['completely_answered']}")

        except Exception as e:
            print(f"Error processing question: {e}")

    # Save results
    output_file = f"results/enhanced_main_rag_hybrid_n{args.n}_a{args.alpha}.json"
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
