#!/usr/bin/env python3
"""
RAG Evaluation Script

Evaluates multiple RAG systems against ground truth using a powerful LLM.
Uses parallel processing for faster evaluation.

Usage:
  python RAG_evaluation.py --rag_files system1.jsonl system2.jsonl system3.jsonl \
                        --ground_truth ground_truth.jsonl \
                        --output evaluation_results \
                        --workers 16
"""

import json
import re
import time
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("rag_evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger("RAG_Evaluator")


class RAGEvaluator:
    def __init__(self, api_key=None):
        """Initialize the RAG evaluator with OpenAI client."""
        # Load API key from file if not provided
        if not api_key:
            path_to_key = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
            if os.path.exists(path_to_key):
                with open(path_to_key) as keyfile:
                    api_key = keyfile.readline().strip()

        if not api_key:
            raise ValueError("No API key provided or found in ~/.scadsai-api-key")

        # Initialize OpenAI client with custom base URL
        self.client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=api_key)

        # Find the Llama model
        self.model = self._find_llama_model()
        logger.info(f"Using model: {self.model}")

    def _find_llama_model(self):
        """Find a Llama model from available models."""
        try:
            models = self.client.models.list().data
            # First try to find Llama-3.3-70B
            for model in models:
                if "llama-3.3-70b" in model.id.lower():
                    return model.id

            # If not found, look for any Llama model
            for model in models:
                if "llama" in model.id.lower():
                    return model.id

            # If no Llama model found, use the first available model
            if models:
                return models[0].id
            else:
                raise ValueError("No models available")
        except Exception as e:
            logger.error(f"Error finding model: {e}")
            return "meta-llama/Llama-3.3-70B-Instruct"  # Fallback

    def evaluate_correctness(self, question, ground_truth_answer, generated_answer):
        """
        Evaluate how well the generated answer matches the ground truth.
        This evaluates coverage and relevance.

        Args:
            question: The question being answered
            ground_truth_answer: The reference answer (ground truth)
            generated_answer: The RAG system's generated answer

        Returns:
            Dictionary with score and full evaluation
        """
        # Truncate to 300 words for evaluation
        words = generated_answer.split()
        if len(words) > 300:
            evaluated_answer = " ".join(words[:300])
            logger.info(f"Answer truncated to first 300 words for evaluation.")
        else:
            evaluated_answer = generated_answer

        prompt = f"""You are an expert evaluator assessing the correctness of an answer to a question.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth_answer}

GENERATED ANSWER: {evaluated_answer}

Evaluate the correctness of the generated answer on a continuous scale from -1 to 2:
- 2: Correct and completely relevant (no irrelevant information)
- 1: Correct but contains irrelevant information
- 0: No answer provided (e.g. "The provided documents do not contain sufficient information")
- -1: Incorrect answer

Consider these aspects:
1. Coverage: What portion of vital information from the ground truth is present in the generated answer?
2. Relevance: Is the generated answer directly addressing the question without unnecessary information?

First, analyze the answer step by step explaining your reasoning.

IMPORTANT: You must end your response with exactly this format:
FINAL_CORRECTNESS_SCORE: [your numeric score]

Example:
FINAL_CORRECTNESS_SCORE: 1.0

YOUR EVALUATION:
"""
        # Call the API using OpenAI client
        result = self._call_api(prompt)
        score = self._extract_correctness_score(result)
        return {"score": score, "full_evaluation": result}

    def evaluate_faithfulness(
        self, question, generated_answer, passages, max_passages=10
    ):
        """
        Evaluate whether the generated answer is grounded in the retrieved passages.

        Args:
            question: The question being answered
            generated_answer: The RAG system's generated answer
            passages: List of retrieved passages used to generate the answer
            max_passages: Maximum number of passages to include in evaluation

        Returns:
            Dictionary with score and full evaluation
        """
        # Truncate to 300 words for evaluation
        words = generated_answer.split()
        if len(words) > 300:
            evaluated_answer = " ".join(words[:300])
        else:
            evaluated_answer = generated_answer

        # Limit to first 10 passages for evaluation
        limited_passages = passages[:max_passages]

        # Format passages for prompt
        formatted_passages = "\n\n".join(
            [f"PASSAGE {i+1}:\n{p['passage']}" for i, p in enumerate(limited_passages)]
        )

        prompt = f"""You are an expert evaluator assessing whether an answer is faithfully grounded in the provided passages.

QUESTION: {question}

GENERATED ANSWER: {evaluated_answer}

RETRIEVED PASSAGES:
{formatted_passages}

Evaluate the faithfulness of the answer on a continuous scale from -1 to 1:
- 1: Full support (all claims in the answer are directly supported by the passages)
- 0: Partial support (some claims are supported, others are not)
- -1: No support (none of the claims are supported by the passages)

First, analyze each claim of the answer and check if it's supported by the passages.

IMPORTANT: You must end your response with exactly this format:
FINAL_FAITHFULNESS_SCORE: [your numeric score]

Example:
FINAL_FAITHFULNESS_SCORE: 0.0

YOUR EVALUATION:
"""
        # Call the API using OpenAI client
        result = self._call_api(prompt)
        score = self._extract_faithfulness_score(result)
        return {"score": score, "full_evaluation": result}

    def _call_api(self, prompt, max_retries=3, retry_delay=5):
        """Call the LLM using OpenAI client with retries."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Lower temperature for more consistent evaluation
                )

                return response.choices[0].message.content

            except Exception as e:
                logger.warning(
                    f"Error calling API (attempt {attempt+1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise

    def _extract_correctness_score(self, evaluation_text: str) -> float:
        """Extract the correctness score from the evaluation text."""
        # 1) Look for the required final score format
        m = re.search(
            r"FINAL_CORRECTNESS_SCORE:\s*(-1(?:\.0+)?|0(?:\.\d+)?|1(?:\.\d+)?|2(?:\.0+)?)",
            evaluation_text,
            re.IGNORECASE,
        )
        if m:
            return float(m.group(1))

        # 2) Fallback to original patterns for backward compatibility
        m = re.search(
            r"final\s+correctness\s+score\s*(?:is|:)?\s*(-1(?:\.0+)?|0(?:\.\d+)?|1(?:\.\d+)?|2(?:\.0+)?)",
            evaluation_text,
            re.IGNORECASE,
        )
        if m:
            return float(m.group(1))

        # 3) Look for score mentions in the text
        m = re.search(
            r"(?:correctness\s+)?score.*?(?:is|:)\s*(-1(?:\.0+)?|0(?:\.\d+)?|1(?:\.\d+)?|2(?:\.0+)?)",
            evaluation_text,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            return float(m.group(1))

        # 4) Last resort: look for standalone numbers on their own lines
        lines = evaluation_text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if re.fullmatch(r"-1(?:\.0+)?|0(?:\.\d+)?|1(?:\.\d+)?|2(?:\.0+)?", line):
                score = float(line)
                if -1 <= score <= 2:
                    return score

        logger.warning(f"Unable to extract correctness score from:\n{evaluation_text}")
        raise ValueError("Could not parse correctness score from evaluation text.")

    def _extract_faithfulness_score(self, evaluation_text: str) -> float:
        """Extract the faithfulness score from the evaluation text."""
        # 1) Look for the required final score format
        m = re.search(
            r"FINAL_FAITHFULNESS_SCORE:\s*(-1(?:\.0+)?|0(?:\.\d+)?|1(?:\.0+)?)",
            evaluation_text,
            re.IGNORECASE,
        )
        if m:
            return float(m.group(1))

        # 2) Fallback to original patterns for backward compatibility
        m = re.search(
            r"final\s+faithfulness\s+score\s*(?:is|:)?\s*(-1(?:\.0+)?|0(?:\.\d+)?|1(?:\.0+)?)",
            evaluation_text,
            re.IGNORECASE,
        )
        if m:
            return float(m.group(1))

        # 3) Look for score mentions in the text
        m = re.search(
            r"(?:faithfulness\s+)?score.*?(?:is|:)\s*(-1(?:\.0+)?|0(?:\.\d+)?|1(?:\.0+)?)",
            evaluation_text,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            return float(m.group(1))

        # 4) Last resort: look for standalone numbers on their own lines
        lines = evaluation_text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if re.fullmatch(r"-1(?:\.0+)?|0(?:\.\d+)?|1(?:\.0+)?", line):
                score = float(line)
                if -1 <= score <= 1:
                    return score

        logger.warning(f"Unable to extract faithfulness score from:\n{evaluation_text}")
        raise ValueError("Could not parse faithfulness score from evaluation text.")

    def evaluate_rag_output(self, rag_data, ground_truth_data):
        """
        Evaluate a single RAG output against ground truth.

        Args:
            rag_data: Dictionary containing RAG system output (question, answer, passages)
            ground_truth_data: Dictionary containing ground truth (question, answer)

        Returns:
            Dictionary with evaluation results
        """
        question = rag_data["question"]
        generated_answer = rag_data["answer"]

        # Get ground truth answer
        ground_truth_answer = ground_truth_data["answer"]

        # Get passages (normalize if needed)
        if "passages" in rag_data:
            passages = rag_data["passages"]
        else:
            # Try to extract passages from alternative formats
            passages = []
            if "context" in rag_data:
                passages = [{"passage": p} for p in rag_data["context"]]

        results = {
            "question": question,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "passages_count": len(passages),
            "system_name": rag_data.get("system_name", "Unknown"),
        }

        # Evaluate faithfulness (how well the answer is grounded in passages)
        logger.info(f"Evaluating faithfulness for '{question[:30]}...'")
        faithfulness = self.evaluate_faithfulness(question, generated_answer, passages)
        results["faithfulness_score"] = faithfulness["score"]
        results["faithfulness_evaluation"] = faithfulness["full_evaluation"]

        # Evaluate correctness (how well the answer matches ground truth)
        logger.info(f"Evaluating correctness for '{question[:30]}...'")
        correctness = self.evaluate_correctness(
            question, ground_truth_answer, generated_answer
        )

        # CRITICAL RULE: If faithfulness is -1 (no support), correctness must be -1
        if results["faithfulness_score"] == -1:
            logger.info(
                f"Faithfulness is -1, forcing correctness to -1 for: '{question[:30]}...'"
            )
            results["correctness_score"] = -1
            results["correctness_evaluation"] = (
                correctness["full_evaluation"]
                + "\n\nNOTE: Correctness score adjusted to -1 because faithfulness score is -1 "
                "(answer is not supported by the provided passages)."
            )
        else:
            results["correctness_score"] = correctness["score"]
            results["correctness_evaluation"] = correctness["full_evaluation"]

        # Calculate combined score
        results["combined_score"] = (
            results["correctness_score"] + results["faithfulness_score"]
        ) / 2

        return results


def evaluate_question(args):
    """Process a single question (for parallel processing)."""
    i, rag_item, system_name, system_dir, ground_truth_item, api_key = args

    # Create a new evaluator instance for thread safety
    evaluator = RAGEvaluator(api_key=api_key)

    question = rag_item["question"]

    try:
        # Add system name
        rag_item["system_name"] = system_name

        # Evaluate question
        eval_results = evaluator.evaluate_rag_output(rag_item, ground_truth_item)

        # Save individual result
        result_file = os.path.join(system_dir, f"question_{i+1}.json")
        with open(result_file, "w") as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"✓ Evaluated {system_name}: '{question[:50]}...'")
        return eval_results

    except Exception as e:
        logger.error(f"✗ Error evaluating {system_name} Q{i+1}: {e}")
        return None


def generate_summary(results, output_file):
    """
    Generate summary statistics from evaluation results.

    Args:
        results: List of evaluation results
        output_file: Path to save summary JSON

    Returns:
        Dictionary with summary statistics
    """
    # Extract scores
    faithfulness_scores = [r["faithfulness_score"] for r in results]
    correctness_scores = [r["correctness_score"] for r in results]
    combined_scores = [r["combined_score"] for r in results]

    # Calculate statistics
    summary = {
        "total_questions": len(results),
        "faithfulness": {
            "mean": float(np.mean(faithfulness_scores)),
            "std": float(np.std(faithfulness_scores)),
            "min": float(min(faithfulness_scores)),
            "max": float(max(faithfulness_scores)),
            "median": float(np.median(faithfulness_scores)),
        },
        "correctness": {
            "mean": float(np.mean(correctness_scores)),
            "std": float(np.std(correctness_scores)),
            "min": float(min(correctness_scores)),
            "max": float(max(correctness_scores)),
            "median": float(np.median(correctness_scores)),
        },
        "combined": {
            "mean": float(np.mean(combined_scores)),
            "std": float(np.std(combined_scores)),
            "min": float(min(combined_scores)),
            "max": float(max(combined_scores)),
            "median": float(np.median(combined_scores)),
        },
        # Additional statistics
        "score_distributions": {
            "faithfulness_negative": sum(1 for s in faithfulness_scores if s == -1),
            "faithfulness_zero": sum(1 for s in faithfulness_scores if s == 0),
            "faithfulness_positive": sum(1 for s in faithfulness_scores if s > 0),
            "correctness_negative": sum(1 for s in correctness_scores if s == -1),
            "correctness_zero": sum(1 for s in correctness_scores if s == 0),
            "correctness_positive": sum(1 for s in correctness_scores if s > 0),
            "forced_correctness_adjustments": sum(
                1
                for r in results
                if r["faithfulness_score"] == -1
                and "adjusted to -1" in r.get("correctness_evaluation", "")
            ),
        },
    }

    # Save summary to file
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Log summary
    logger.info(f"Summary statistics saved to {output_file}")
    logger.info(
        f"Faithfulness: {summary['faithfulness']['mean']:.2f} ± {summary['faithfulness']['std']:.2f}"
    )
    logger.info(
        f"Correctness: {summary['correctness']['mean']:.2f} ± {summary['correctness']['std']:.2f}"
    )
    logger.info(
        f"Combined: {summary['combined']['mean']:.2f} ± {summary['combined']['std']:.2f}"
    )
    logger.info(
        f"Forced correctness adjustments: {summary['score_distributions']['forced_correctness_adjustments']}"
    )

    return summary


def generate_summary_tables(results, output_dir):
    """
    Generate detailed CSV tables from evaluation results.

    Args:
        results: List of evaluation results
        output_dir: Directory to save CSV files
    """
    # Create detailed results table
    details = []
    for r in results:
        details.append(
            {
                "question": r["question"],
                "faithfulness_score": r["faithfulness_score"],
                "correctness_score": r["correctness_score"],
                "combined_score": r["combined_score"],
                "passages_count": r["passages_count"],
                "forced_adjustment": (
                    "Yes"
                    if (
                        r["faithfulness_score"] == -1
                        and "adjusted to -1" in r.get("correctness_evaluation", "")
                    )
                    else "No"
                ),
            }
        )

    # Save to CSV
    details_df = pd.DataFrame(details)
    details_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)

    # Create score distribution table
    score_ranges = {
        "faithfulness": [
            {"range": "[-1.0, -0.5)", "count": 0},
            {"range": "[-0.5, 0.0)", "count": 0},
            {"range": "[0.0, 0.5)", "count": 0},
            {"range": "[0.5, 1.0]", "count": 0},
        ],
        "correctness": [
            {"range": "[-1.0, -0.5)", "count": 0},
            {"range": "[-0.5, 0.0)", "count": 0},
            {"range": "[0.0, 0.5)", "count": 0},
            {"range": "[0.5, 1.0)", "count": 0},
            {"range": "[1.0, 1.5)", "count": 0},
            {"range": "[1.5, 2.0]", "count": 0},
        ],
    }

    # Count scores in each range
    for r in results:
        f_score = r["faithfulness_score"]
        c_score = r["correctness_score"]

        # Faithfulness
        if -1.0 <= f_score < -0.5:
            score_ranges["faithfulness"][0]["count"] += 1
        elif -0.5 <= f_score < 0.0:
            score_ranges["faithfulness"][1]["count"] += 1
        elif 0.0 <= f_score < 0.5:
            score_ranges["faithfulness"][2]["count"] += 1
        elif 0.5 <= f_score <= 1.0:
            score_ranges["faithfulness"][3]["count"] += 1

        # Correctness
        if -1.0 <= c_score < -0.5:
            score_ranges["correctness"][0]["count"] += 1
        elif -0.5 <= c_score < 0.0:
            score_ranges["correctness"][1]["count"] += 1
        elif 0.0 <= c_score < 0.5:
            score_ranges["correctness"][2]["count"] += 1
        elif 0.5 <= c_score < 1.0:
            score_ranges["correctness"][3]["count"] += 1
        elif 1.0 <= c_score < 1.5:
            score_ranges["correctness"][4]["count"] += 1
        elif 1.5 <= c_score <= 2.0:
            score_ranges["correctness"][5]["count"] += 1

    # Save distribution tables
    faithfulness_df = pd.DataFrame(score_ranges["faithfulness"])
    faithfulness_df.to_csv(
        os.path.join(output_dir, "faithfulness_distribution.csv"), index=False
    )

    correctness_df = pd.DataFrame(score_ranges["correctness"])
    correctness_df.to_csv(
        os.path.join(output_dir, "correctness_distribution.csv"), index=False
    )


def generate_cross_system_comparison(system_summaries, output_dir):
    """
    Generate comparison table across multiple RAG systems.

    Args:
        system_summaries: Dictionary mapping system names to their summaries
        output_dir: Directory to save comparison files
    """
    # Extract summary metrics for each system
    comparison_data = {
        "System": [],
        "Faithfulness": [],
        "Faithfulness_StdDev": [],
        "Correctness": [],
        "Correctness_StdDev": [],
        "Combined": [],
        "Combined_StdDev": [],
        "Forced_Adjustments": [],
    }

    for system_name, summary in system_summaries.items():
        comparison_data["System"].append(system_name)
        comparison_data["Faithfulness"].append(summary["faithfulness"]["mean"])
        comparison_data["Faithfulness_StdDev"].append(summary["faithfulness"]["std"])
        comparison_data["Correctness"].append(summary["correctness"]["mean"])
        comparison_data["Correctness_StdDev"].append(summary["correctness"]["std"])
        comparison_data["Combined"].append(summary["combined"]["mean"])
        comparison_data["Combined_StdDev"].append(summary["combined"]["std"])
        comparison_data["Forced_Adjustments"].append(
            summary["score_distributions"]["forced_correctness_adjustments"]
        )

    # Create DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, "system_comparison.csv"), index=False)

    # Create JSON version with more details
    comparison_json = {
        "timestamp": datetime.now().isoformat(),
        "systems": list(system_summaries.keys()),
        "metrics": {
            "faithfulness": {
                system: summary["faithfulness"]
                for system, summary in system_summaries.items()
            },
            "correctness": {
                system: summary["correctness"]
                for system, summary in system_summaries.items()
            },
            "combined": {
                system: summary["combined"]
                for system, summary in system_summaries.items()
            },
        },
    }

    with open(os.path.join(output_dir, "system_comparison.json"), "w") as f:
        json.dump(comparison_json, f, indent=2)

    # Print comparison table
    logger.info("\n=== CROSS-SYSTEM COMPARISON ===")
    logger.info(
        f"System Comparison saved to {os.path.join(output_dir, 'system_comparison.csv')}"
    )
    logger.info("\nSystem\tFaithfulness\tCorrectness\tCombined\tForced_Adj")
    for i, system in enumerate(comparison_data["System"]):
        logger.info(
            f"{system}\t{comparison_data['Faithfulness'][i]:.2f}\t"
            f"{comparison_data['Correctness'][i]:.2f}\t"
            f"{comparison_data['Combined'][i]:.2f}\t"
            f"{comparison_data['Forced_Adjustments'][i]}"
        )

    # Create summary plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Create bar chart comparing systems
        plt.figure(figsize=(12, 8))
        systems = comparison_data["System"]
        x = np.arange(len(systems))
        width = 0.25

        # Plot bars
        plt.bar(x - width, comparison_data["Faithfulness"], width, label="Faithfulness")
        plt.bar(x, comparison_data["Correctness"], width, label="Correctness")
        plt.bar(x + width, comparison_data["Combined"], width, label="Combined")

        # Add labels and legend
        plt.xlabel("System")
        plt.ylabel("Score")
        plt.title("RAG System Comparison")
        plt.xticks(x, systems)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save plot
        plt.savefig(
            os.path.join(output_dir, "system_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        logger.info(
            f"Comparison plot saved to {os.path.join(output_dir, 'system_comparison.png')}"
        )
    except Exception as e:
        logger.warning(f"Error creating plot: {e}")


def evaluate_systems(
    rag_files,
    ground_truth_file,
    output_dir,
    api_key=None,
    sample_size=None,
    max_workers=8,
):
    """
    Evaluate multiple RAG systems against ground truth in parallel.

    Args:
        rag_files: List of paths to JSONL files for each RAG system
        ground_truth_file: Path to ground truth JSONL file
        output_dir: Directory to save results
        api_key: Optional API key
        sample_size: Optional number of questions to sample (for testing)
        max_workers: Maximum number of parallel workers
    """
    # Create timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"eval_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {run_dir}")

    # Load ground truth data
    ground_truth_data = []
    with open(ground_truth_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    ground_truth_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing ground truth line: {e}")

    # Create ground truth lookup by question
    ground_truth_lookup = {item["question"]: item for item in ground_truth_data}
    logger.info(f"Loaded {len(ground_truth_lookup)} ground truth questions")

    # Process each RAG system
    system_summaries = {}

    for rag_file in rag_files:
        system_name = os.path.basename(rag_file).replace(".jsonl", "")
        logger.info(f"\nEvaluating system: {system_name}")

        # Create directory for this system
        system_dir = os.path.join(run_dir, system_name)
        os.makedirs(system_dir, exist_ok=True)

        # Load RAG data
        rag_data = []
        with open(rag_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        rag_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing RAG data line: {e}")

        logger.info(f"Loaded {len(rag_data)} questions for {system_name}")

        # Sample if requested
        if sample_size and sample_size < len(rag_data):
            import random

            random.seed(42)  # For reproducibility
            rag_data = random.sample(rag_data, sample_size)
            logger.info(f"Sampled {sample_size} questions for evaluation")

        # Prepare parallel tasks
        tasks = []
        for i, rag_item in enumerate(rag_data):
            question = rag_item["question"]
            if question in ground_truth_lookup:
                tasks.append(
                    (
                        i,
                        rag_item,
                        system_name,
                        system_dir,
                        ground_truth_lookup[question],
                        api_key,
                    )
                )
            else:
                logger.warning(f"No matching ground truth for question: {question}")

        logger.info(
            f"Processing {len(tasks)} questions with {max_workers} parallel workers..."
        )

        # Process in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list_of_futures = []
            for task in tasks:
                future = executor.submit(evaluate_question, task)
                list_of_futures.append(future)

            # Create a progress bar
            for future in tqdm(list_of_futures, desc=f"Evaluating {system_name}"):
                result = future.result()
                if result:
                    results.append(result)

        # Generate summary for this system
        if results:
            system_summary = generate_summary(
                results, os.path.join(system_dir, "summary.json")
            )
            generate_summary_tables(results, system_dir)
            system_summaries[system_name] = system_summary

    # Generate cross-system comparison
    if len(system_summaries) > 1:
        generate_cross_system_comparison(system_summaries, run_dir)

    return system_summaries


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG systems against ground truth"
    )
    parser.add_argument(
        "--rag_files", nargs="+", required=True, help="Paths to RAG system JSONL files"
    )
    parser.add_argument(
        "--ground_truth", required=True, help="Path to ground truth JSONL file"
    )
    parser.add_argument(
        "--output", default="evaluation_results", help="Directory to save results"
    )
    parser.add_argument(
        "--api_key",
        help="API key for SCADS.AI (optional if stored in ~/.scadsai-api-key)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Evaluate only a sample of questions (specify number)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel workers (default: 8)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Log start time
    start_time = time.time()
    logger.info(f"Starting evaluation with {args.workers} workers")

    # Evaluate systems
    try:
        evaluate_systems(
            args.rag_files,
            args.ground_truth,
            args.output,
            args.api_key,
            args.sample,
            args.workers,
        )

        # Log end time and duration
        end_time = time.time()
        duration = end_time - start_time
        logger.info(
            f"Evaluation completed in {duration:.2f} seconds ({duration/60:.2f} minutes)"
        )

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
