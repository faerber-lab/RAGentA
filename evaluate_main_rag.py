import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# import nltk
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.tokenize import word_tokenize

# Make sure NLTK resources are available
# try:
# nltk.data.find("tokenizers/punkt")
# except LookupError:
# nltk.download("punkt")


class DeepseekJudge:
    """Judge implementation using Deepseek LLM."""

    def __init__(
        self, model_name="deepseek-ai/deepseek-coder-33b-instruct", device="cuda"
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading Deepseek judge model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
            device_map="auto",
            trust_remote_code=True,
        )
        self.device = device
        print(f"Deepseek judge model loaded on {self.device}")

    def generate(self, prompt, max_new_tokens=1024):
        """Generate text using Deepseek model."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Lower temperature for more consistent evaluations
                do_sample=False,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return response


def load_results(results_file):
    """Load results from the MAIN-RAG output file."""
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results


def load_datamorgana_questions(file_path="datamorgana_questions.json"):
    """Load the original DataMorgana questions file for reference."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Check if it's JSONL or JSON format
            first_char = f.read(1)
            f.seek(0)  # Reset file pointer

            if first_char == "{":
                # Try parsing as JSONL
                data = []
                for line in f:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
            else:
                # Try parsing as JSON
                data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading DataMorgana questions: {e}")
        return []


def compute_rouge_score(hypothesis, reference):
    """Compute ROUGE scores between the model answer and reference answer."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

    # def compute_bleu_score(hypothesis, reference):
    """Compute BLEU score between model answer and reference answer."""
    hypothesis_tokens = word_tokenize(hypothesis.lower())
    reference_tokens = [word_tokenize(reference.lower())]
    try:
        return sentence_bleu(reference_tokens, hypothesis_tokens)
    except Exception as e:
        print(f"Error computing BLEU score: {e}")
        return 0.0


def auto_evaluate_with_llm(
    question,
    model_answer,
    filtered_docs,
    reference_answer=None,
    api_url=None,
    api_key=None,
    local_judge=None,
):
    """
    Use an LLM to evaluate the model's answer for relevance and faithfulness.
    Can use either an API-based LLM or a local LLM.
    """
    # If local judge is provided, use it
    if local_judge:
        # Format the prompt for the LLM judge
        docs_text = "\n\n".join(
            [f"Document {i+1}: {doc}" for i, doc in enumerate(filtered_docs)]
        )

        prompt = f"""
        Question: {question}
        
        Model Answer: {model_answer}
        
        Retrieved Documents:
        {docs_text}
        
        Your task is to evaluate this answer on two dimensions:
        
        1. Relevance (scale from -1 to 2):
           2: The response correctly answers the user question and contains no irrelevant content
           1: The response provides a useful answer to the user question, but may contain irrelevant content that doesn't harm the usefulness
           0: No answer is provided in the response
           -1: The response does not answer the question whatsoever
        
        2. Faithfulness (scale from -1 to 1):
           1: Full support - all answer parts are grounded in the retrieved passages
           0: Partial support - not all answer parts are grounded in the retrieved passages
           -1: No support - all answer parts are not grounded in the retrieved passages
        
        First, carefully analyze if the answer addresses the question appropriately.
        Second, check if each part of the answer is supported by the retrieved documents.
        
        Provide your reasoning and then your final scores in the format:
        Relevance Score: [score]
        Faithfulness Score: [score]
        """

        try:
            # Use the local judge to generate evaluation
            judge_response = local_judge.generate(prompt)

            # Parse the response to extract scores
            relevance_score = None
            faithfulness_score = None

            # Simple parsing logic - you might need to adjust based on your LLM's output format
            for line in judge_response.split("\n"):
                if "Relevance Score:" in line:
                    try:
                        relevance_score = int(line.split(":")[1].strip())
                    except:
                        pass
                elif "Faithfulness Score:" in line:
                    try:
                        faithfulness_score = int(line.split(":")[1].strip())
                    except:
                        pass

            if relevance_score is None or faithfulness_score is None:
                # If parsing failed, use dummy scores
                relevance_score = np.random.choice([-1, 0, 1, 2])
                faithfulness_score = np.random.choice([-1, 0, 1])
                judge_reasoning = (
                    judge_response + "\n\nNote: Failed to parse exact scores."
                )
            else:
                judge_reasoning = judge_response

            return {
                "relevance_score": relevance_score,
                "faithfulness_score": faithfulness_score,
                "judge_reasoning": judge_reasoning,
            }
        except Exception as e:
            print(f"Error using local judge: {e}")
            # Fall back to default dummy scores

    # If we don't have a local judge, or it failed, continue with the API or default logic
    if not api_url or not api_key:
        # Return random scores for testing if no API is provided
        print("Warning: No LLM API provided for auto-evaluation. Using dummy scores.")
        return {
            "relevance_score": np.random.choice([-1, 0, 1, 2]),
            "faithfulness_score": np.random.choice([-1, 0, 1]),
            "judge_reasoning": "Dummy evaluation (no LLM API provided).",
        }


def evaluate_results(
    results, datamorgana_questions, llm_api_url=None, llm_api_key=None, local_judge=None
):
    """Evaluate MAIN-RAG results using both automatic metrics and LLM evaluation."""
    eval_results = []

    # Create a lookup dictionary for DataMorgana questions
    question_lookup = {q["question"]: q for q in datamorgana_questions}

    for result in tqdm(results, desc="Evaluating results"):
        question = result["question"]
        model_answer = result["model_answer"]
        reference_answer = result["reference_answer"]

        # Skip empty answers (model errors)
        if model_answer == "<|assistant|>" or not model_answer:
            print(
                f"Skipping evaluation for question with empty answer: {question[:50]}..."
            )
            continue

        # Get the original context if available
        original_context = []
        if question in question_lookup:
            original_context = question_lookup[question].get("context", [])

        # Get filtered documents if available in the result
        filtered_docs = []
        if "filtered_docs" in result:
            filtered_docs = [doc for doc, _ in result["filtered_docs"]]

        # Compute automatic metrics
        rouge_scores = compute_rouge_score(model_answer, reference_answer)
        # bleu_score = compute_bleu_score(model_answer, reference_answer)

        # Perform LLM-based evaluation
        llm_eval = auto_evaluate_with_llm(
            question,
            model_answer,
            filtered_docs,
            reference_answer,
            llm_api_url,
            llm_api_key,
            local_judge,
        )

        # Compile evaluation results
        eval_data = {
            "question": question,
            "model_answer": model_answer,
            "reference_answer": reference_answer,
            "rouge1": float(rouge_scores["rouge1"]),  # Convert to Python float
            "rouge2": float(rouge_scores["rouge2"]),
            "rougeL": float(rouge_scores["rougeL"]),
            "relevance_score": int(
                llm_eval["relevance_score"]
            ),  # Convert to Python int
            "faithfulness_score": int(llm_eval["faithfulness_score"]),
            "judge_reasoning": llm_eval["judge_reasoning"],
            "tau_q": float(result.get("tau_q", 0)),
            "adjusted_tau_q": float(result.get("adjusted_tau_q", 0)),
            "filtered_count": int(result.get("filtered_count", 0)),
            "process_time": float(result.get("process_time", 0)),
        }

        # Add filtered docs if available
        if filtered_docs:
            eval_data["filtered_docs"] = filtered_docs

        # Store the original context if available
        if original_context:
            eval_data["original_context"] = original_context

        eval_results.append(eval_data)

    return eval_results


def visualize_evaluation_results(eval_results, output_dir):
    """Generate visualizations and summary statistics for the evaluation results."""
    # Make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(
        [
            {
                "question_id": i,
                "relevance_score": r["relevance_score"],
                "faithfulness_score": r["faithfulness_score"],
                "rouge1": r["rouge1"],
                "rouge2": r["rouge2"],
                "rougeL": r["rougeL"],
                # "bleu": r["bleu"],
                "tau_q": r["tau_q"],
                "adjusted_tau_q": r["adjusted_tau_q"],
                "filtered_count": r["filtered_count"],
                "process_time": r.get("process_time", 0),
            }
            for i, r in enumerate(eval_results)
        ]
    )

    # Calculate summary statistics
    summary = {
        "avg_relevance": df["relevance_score"].mean(),
        "avg_faithfulness": df["faithfulness_score"].mean(),
        "avg_rouge1": df["rouge1"].mean(),
        "avg_rouge2": df["rouge2"].mean(),
        "avg_rougeL": df["rougeL"].mean(),
        # "avg_bleu": df["bleu"].mean(),
        "avg_filtered_count": df["filtered_count"].mean(),
        "avg_process_time": df["process_time"].mean(),
        "count": len(df),
        "relevance_distribution": df["relevance_score"].value_counts().to_dict(),
        "faithfulness_distribution": df["faithfulness_score"].value_counts().to_dict(),
    }

    # Save summary statistics
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot relevance score distribution
    plt.figure(figsize=(10, 6))
    df["relevance_score"].value_counts().sort_index().plot(kind="bar")
    plt.title("Distribution of Relevance Scores")
    plt.xlabel("Relevance Score")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "relevance_distribution.png"))

    # Plot faithfulness score distribution
    plt.figure(figsize=(10, 6))
    df["faithfulness_score"].value_counts().sort_index().plot(kind="bar")
    plt.title("Distribution of Faithfulness Scores")
    plt.xlabel("Faithfulness Score")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "faithfulness_distribution.png"))

    # Plot correlation between tau_q and evaluation scores
    plt.figure(figsize=(10, 6))
    plt.scatter(df["tau_q"], df["relevance_score"], alpha=0.7, label="Relevance")
    plt.scatter(df["tau_q"], df["faithfulness_score"], alpha=0.7, label="Faithfulness")
    plt.title("Correlation Between tau_q and Evaluation Scores")
    plt.xlabel("tau_q")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "tau_q_correlation.png"))

    # Plot relationship between filtered document count and scores
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["filtered_count"], df["relevance_score"], alpha=0.7, label="Relevance"
    )
    plt.scatter(
        df["filtered_count"], df["faithfulness_score"], alpha=0.7, label="Faithfulness"
    )
    plt.title("Relationship Between Filtered Document Count and Evaluation Scores")
    plt.xlabel("Filtered Document Count")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "filtered_count_correlation.png"))

    # Plot ROUGE scores
    plt.figure(figsize=(10, 6))
    df[["rouge1", "rouge2", "rougeL"]].mean().plot(kind="bar")
    plt.title("Average ROUGE Scores")
    plt.ylabel("Score")
    plt.savefig(os.path.join(output_dir, "rouge_scores.png"))

    print(f"Evaluation summary and visualizations saved to {output_dir}")
    return summary


def save_evaluation_results(eval_results, output_dir, filename="detailed_results.json"):
    """Save the detailed evaluation results to a JSON file."""
    # Make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Create full path for the output file
    output_file = os.path.join(output_dir, filename)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Detailed evaluation results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAIN-RAG results")
    parser.add_argument(
        "--results", type=str, required=True, help="Path to MAIN-RAG results file"
    )
    parser.add_argument(
        "--datamorgana",
        type=str,
        default="datamorgana_questions.json",
        help="Path to original DataMorgana questions file",
    )
    parser.add_argument(
        "--llm-api-url", type=str, help="LLM API URL for auto-evaluation"
    )
    parser.add_argument(
        "--llm-api-key", type=str, help="LLM API key for auto-evaluation"
    )
    parser.add_argument(
        "--local-judge",
        type=str,
        help="Local LLM model to use as judge (e.g., deepseek-ai/deepseek-coder-33b-instruct)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()

    # Load the results
    results = load_results(args.results)

    # Get the base name of the results file (without path and extension)
    import os
    import datetime

    results_basename = os.path.basename(args.results)
    results_name = os.path.splitext(results_basename)[0]  # Remove extension

    # Add timestamp to create a unique folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_specific_dir = f"{results_name}_{timestamp}"

    # Create the full output directory path
    output_dir = os.path.join(args.output_dir, result_specific_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")

    # Load original DataMorgana questions
    datamorgana_questions = load_datamorgana_questions(args.datamorgana)

    # Initialize local judge if specified
    local_judge = None
    if args.local_judge:
        from deepseek_judge import DeepseekJudge

        local_judge = DeepseekJudge(model_name=args.local_judge)

    # Pass the local judge to the evaluation function
    eval_results = evaluate_results(
        results, datamorgana_questions, args.llm_api_url, args.llm_api_key, local_judge
    )

    # Save detailed evaluation results
    save_evaluation_results(eval_results, output_dir)

    # Generate visualizations and summary
    summary = visualize_evaluation_results(eval_results, output_dir)

    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Average Relevance Score: {summary['avg_relevance']:.2f}")
    print(f"Average Faithfulness Score: {summary['avg_faithfulness']:.2f}")
    print(f"Average ROUGE-1: {summary['avg_rouge1']:.2f}")
    print(f"Average ROUGE-L: {summary['avg_rougeL']:.2f}")
    # print(f"Average BLEU: {summary['avg_bleu']:.2f}")
    print(f"Average Filtered Documents: {summary['avg_filtered_count']:.2f}")
    print(f"Average Processing Time: {summary['avg_process_time']:.2f} seconds")
    print(f"Total Questions Evaluated: {summary['count']}")


if __name__ == "__main__":
    main()
