import json
import re
import time
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from datetime import datetime

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
        self.client = OpenAI(
            base_url="https://llm.scads.ai/v1",
            api_key=api_key
        )
        
        # Find the Llama model
        self.model = self._find_llama_model()
        print(f"Using model: {self.model}")
    
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
            print(f"Error finding model: {e}")
            return "meta-llama/Llama-3.3-70B-Instruct"  # Fallback
        
    def normalize_data(self, data):
        """Normalize different data formats to a standard format."""
        normalized = {"question": "", "answer": "", "passages": []}
        
        # Extract question
        if "question" in data:
            normalized["question"] = data["question"]
        
        # Extract answer
        if "answer" in data:
            normalized["answer"] = data["answer"]
        
        # Extract passages
        if "passages" in data:
            # RAG system format
            normalized["passages"] = data["passages"]
        elif "context" in data:
            # Baseline format - convert context array to passages format
            doc_id = "unknown"
            if "document_ids" in data and data["document_ids"]:
                doc_id = data["document_ids"][0]
            
            normalized["passages"] = [{"passage": p, "doc_IDs": [doc_id]} 
                                    for p in data["context"]]
        
        return normalized
        
    def evaluate_correctness(self, question, reference_answer, generated_answer):
        """Evaluate the correctness of the generated answer."""
        # Truncate to 300 words for evaluation
        words = generated_answer.split()
        if len(words) > 300:
            evaluated_answer = " ".join(words[:300])
        else:
            evaluated_answer = generated_answer
            
        prompt = f"""You are an expert evaluator assessing the correctness of an answer to a question.

QUESTION: {question}

REFERENCE ANSWER: {reference_answer}

GENERATED ANSWER: {evaluated_answer}

Evaluate the correctness of the generated answer on a continuous scale from -1 to 2:
- 2: Correct and completely relevant (no irrelevant information)
- 1: Correct but contains some irrelevant information
- 0: No answer provided (abstention)
- -1: Incorrect answer

Consider these aspects:
1. Coverage: What portion of vital information from the reference is present in the generated answer?
2. Relevance: Is the generated answer directly addressing the question without unnecessary information?

First, analyze the answer step by step. Then provide your final score as a single number between -1 and 2.

YOUR EVALUATION:
"""
        # Call the API using OpenAI client
        result = self._call_api(prompt)
        score = self._extract_correctness_score(result)
        return {
            "score": score,
            "full_evaluation": result
        }
    
    def evaluate_faithfulness(self, question, generated_answer, passages, max_passages=10):
        """Evaluate the faithfulness of the generated answer to the retrieved passages."""
        # Truncate to 300 words for evaluation
        words = generated_answer.split()
        if len(words) > 300:
            evaluated_answer = " ".join(words[:300])
        else:
            evaluated_answer = generated_answer
            
        # Limit to first 10 passages for evaluation
        limited_passages = passages[:max_passages]
        
        # Format passages for prompt
        formatted_passages = "\n\n".join([f"PASSAGE {i+1}:\n{p['passage']}" for i, p in enumerate(limited_passages)])
        
        prompt = f"""You are an expert evaluator assessing whether an answer is faithfully grounded in the provided passages.

QUESTION: {question}

GENERATED ANSWER: {evaluated_answer}

RETRIEVED PASSAGES:
{formatted_passages}

Evaluate the faithfulness of the answer on a continuous scale from -1 to 1:
- 1: Full support (all claims in the answer are directly supported by the passages)
- 0: Partial support (some claims are supported, others are not)
- -1: No support (none of the claims are supported by the passages)

First, analyze each claim in the answer and check if it's supported by the passages.
Then provide your final faithfulness score as a single number between -1 and 1.

YOUR EVALUATION:
"""
        # Call the API using OpenAI client
        result = self._call_api(prompt)
        score = self._extract_faithfulness_score(result)
        return {
            "score": score,
            "full_evaluation": result
        }
    
    def _call_api(self, prompt, max_retries=3, retry_delay=5):
        """Call the LLM using OpenAI client with retries."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1  # Lower temperature for more consistent evaluation
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"Error calling API (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise
    
    def _extract_correctness_score(self, evaluation_text):
        """Extract the correctness score from the evaluation text."""
        # Look for a number between -1 and 2
        matches = re.findall(r"score(?:\s*)?(?:is|:)?(?:\s*)?(-?[0-9.]+)", evaluation_text.lower())
        if matches:
            try:
                score = float(matches[0])
                # Ensure score is within valid range
                return max(min(score, 2), -1)
            except ValueError:
                pass
        
        # If no direct score format, look for the final score statement
        matches = re.findall(r"(-?[0-9.]+)(?:\s*)/(?:\s*)(2)", evaluation_text)
        if matches:
            try:
                numerator, denominator = float(matches[0][0]), float(matches[0][1])
                return numerator * 2 / denominator  # Scale to -1 to 2 range
            except (ValueError, IndexError):
                pass
        
        # Final fallback - look for any number between -1 and 2 in the text
        scores = re.findall(r"(-?[0-9.]+)", evaluation_text)
        for potential_score in scores:
            try:
                score = float(potential_score)
                if -1 <= score <= 2:
                    return score
            except ValueError:
                continue
        
        # If all else fails, estimate from keywords
        if "incorrect" in evaluation_text.lower():
            return -1
        elif "no answer" in evaluation_text.lower() or "abstention" in evaluation_text.lower():
            return 0
        elif "correct but" in evaluation_text.lower() or "partially" in evaluation_text.lower():
            return 1
        elif "correct and" in evaluation_text.lower() or "fully correct" in evaluation_text.lower():
            return 2
        
        # Default to 0 if unable to determine
        print(f"Warning: Unable to extract correctness score from:\n{evaluation_text}")
        return 0
    
    def _extract_faithfulness_score(self, evaluation_text):
        """Extract the faithfulness score from the evaluation text."""
        # Look for a number between -1 and 1
        matches = re.findall(r"score(?:\s*)?(?:is|:)?(?:\s*)?(-?[0-9.]+)", evaluation_text.lower())
        if matches:
            try:
                score = float(matches[0])
                # Ensure score is within valid range
                return max(min(score, 1), -1)
            except ValueError:
                pass
        
        # If no direct score format, look for statements with scores
        if "full support" in evaluation_text.lower():
            return 1
        elif "partial support" in evaluation_text.lower():
            return 0
        elif "no support" in evaluation_text.lower():
            return -1
        
        # Final fallback - look for any number between -1 and 1 in the text
        scores = re.findall(r"(-?[0-9.]+)", evaluation_text)
        for potential_score in scores:
            try:
                score = float(potential_score)
                if -1 <= score <= 1:
                    return score
            except ValueError:
                continue
        
        # Default to 0 if unable to determine
        print(f"Warning: Unable to extract faithfulness score from:\n{evaluation_text}")
        return 0
    
    def evaluate_rag_output(self, rag_data, reference_data=None):
        """Evaluate a RAG output against a reference."""
        # Normalize both data formats
        rag_normalized = self.normalize_data(rag_data)
        reference_normalized = None
        if reference_data:
            reference_normalized = self.normalize_data(reference_data)
        
        question = rag_normalized["question"]
        generated_answer = rag_normalized["answer"]
        passages = rag_normalized["passages"]
        
        # Get reference answer if available
        reference_answer = None
        if reference_normalized:
            reference_answer = reference_normalized["answer"]
        
        results = {
            "question": question,
            "generated_answer": generated_answer,
            "passages_count": len(passages),
            "system_name": rag_data.get("system_name", "Unknown")
        }
        
        # Evaluate faithfulness
        print(f"Evaluating faithfulness...")
        faithfulness = self.evaluate_faithfulness(question, generated_answer, passages)
        results["faithfulness_score"] = faithfulness["score"]
        results["faithfulness_evaluation"] = faithfulness["full_evaluation"]
        
        # Evaluate correctness if reference answer available
        if reference_answer:
            print(f"Evaluating correctness...")
            correctness = self.evaluate_correctness(question, reference_answer, generated_answer)
            results["correctness_score"] = correctness["score"]
            results["correctness_evaluation"] = correctness["full_evaluation"]
            
            # Calculate combined score
            results["combined_score"] = (results["correctness_score"] + results["faithfulness_score"]) / 2
        else:
            print("Skipping correctness evaluation (no reference answer)")
            results["correctness_score"] = None
            results["correctness_evaluation"] = None
            results["combined_score"] = None
        
        return results

def evaluate_multiple_systems(rag_files, baseline_file, output_dir, api_key=None):
    """
    Evaluate multiple RAG systems against a single baseline.
    
    Args:
        rag_files: List of paths to JSONL files for each RAG system
        baseline_file: Path to baseline JSONL file
        output_dir: Directory to save results
        api_key: Optional API key
    """
    # Create timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"eval_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")
    
    # Load baseline data
    baseline_data = []
    with open(baseline_file, 'r') as f:
        for line in f:
            if line.strip():
                baseline_data.append(json.loads(line))
    
    # Create baseline lookup by question
    baseline_lookup = {item["question"]: item for item in baseline_data}
    print(f"Loaded {len(baseline_lookup)} baseline questions")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(api_key=api_key)
    
    # Process each RAG system
    system_results = {}
    
    for rag_file in rag_files:
        system_name = os.path.basename(rag_file).replace(".jsonl", "")
        print(f"\nEvaluating system: {system_name}")
        
        # Create directory for this system
        system_dir = os.path.join(run_dir, system_name)
        os.makedirs(system_dir, exist_ok=True)
        
        # Load RAG data
        rag_data = []
        with open(rag_file, 'r') as f:
            for line in f:
                if line.strip():
                    rag_data.append(json.loads(line))
        
        print(f"Loaded {len(rag_data)} questions for {system_name}")
        
        # Evaluate each question
        results = []
        
        for i, rag_item in enumerate(tqdm(rag_data, desc=f"Evaluating {system_name}")):
            question = rag_item["question"]
            
            # Find matching baseline question
            if question in baseline_lookup:
                baseline_item = baseline_lookup[question]
                
                # Add system name
                rag_item["system_name"] = system_name
                baseline_item["system_name"] = "Baseline"
                
                try:
                    # Evaluate question
                    rag_results = evaluator.evaluate_rag_output(rag_item, baseline_item)
                    baseline_results = evaluator.evaluate_rag_output(baseline_item, baseline_item)
                    
                    # Save individual result
                    result_file = os.path.join(system_dir, f"question_{i+1}.json")
                    
                    # Combine results
                    comparison = {
                        "question": question,
                        "rag_system": {
                            "faithfulness_score": rag_results["faithfulness_score"],
                            "correctness_score": rag_results["correctness_score"],
                            "combined_score": rag_results["combined_score"],
                            "passages_count": rag_results["passages_count"]
                        },
                        "baseline": {
                            "faithfulness_score": baseline_results["faithfulness_score"],
                            "correctness_score": baseline_results["correctness_score"],
                            "combined_score": baseline_results["combined_score"],
                            "passages_count": baseline_results["passages_count"]
                        },
                        "detailed_evaluations": {
                            "rag_system": {
                                "faithfulness_evaluation": rag_results["faithfulness_evaluation"],
                                "correctness_evaluation": rag_results.get("correctness_evaluation")
                            },
                            "baseline": {
                                "faithfulness_evaluation": baseline_results["faithfulness_evaluation"],
                                "correctness_evaluation": baseline_results.get("correctness_evaluation")
                            }
                        }
                    }
                    
                    with open(result_file, 'w') as f:
                        json.dump(comparison, f, indent=2)
                    
                    results.append(comparison)
                    
                except Exception as e:
                    print(f"Error evaluating question {i+1}: {e}")
            else:
                print(f"Warning: No matching baseline for question: {question}")
        
        # Generate summary for this system
        if results:
            system_summary = generate_summary(results, os.path.join(system_dir, "summary.json"))
            generate_summary_tables(results, system_dir)
            system_results[system_name] = system_summary
    
    # Generate cross-system comparison
    if system_results:
        generate_cross_system_comparison(system_results, run_dir)
    
    return system_results

def generate_cross_system_comparison(system_results, output_dir):
    """Generate comparison table across multiple RAG systems."""
    # Extract summary metrics for each system
    comparison_data = {
        "System": [],
        "Faithfulness": [],
        "Faithfulness_StdDev": [],
        "Correctness": [],
        "Correctness_StdDev": [],
        "Combined": [],
        "Combined_StdDev": [],
        "Faithfulness_vs_Baseline": [],
        "Correctness_vs_Baseline": [],
        "Combined_vs_Baseline": []
    }
    
    for system_name, summary in system_results.items():
        comparison_data["System"].append(system_name)
        comparison_data["Faithfulness"].append(summary["rag_system"]["faithfulness"]["mean"])
        comparison_data["Faithfulness_StdDev"].append(summary["rag_system"]["faithfulness"]["std"])
        comparison_data["Correctness"].append(summary["rag_system"]["correctness"]["mean"])
        comparison_data["Correctness_StdDev"].append(summary["rag_system"]["correctness"]["std"])
        comparison_data["Combined"].append(summary["rag_system"]["combined"]["mean"])
        comparison_data["Combined_StdDev"].append(summary["rag_system"]["combined"]["std"])
        comparison_data["Faithfulness_vs_Baseline"].append(summary["comparison"]["faithfulness_diff"])
        comparison_data["Correctness_vs_Baseline"].append(summary["comparison"]["correctness_diff"])
        comparison_data["Combined_vs_Baseline"].append(summary["comparison"]["combined_diff"])
    
    # Create DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, "system_comparison.csv"), index=False)
    
    # Print comparison table
    print("\n=== CROSS-SYSTEM COMPARISON ===")
    print(comparison_df[["System", "Faithfulness", "Correctness", "Combined", "Combined_vs_Baseline"]].to_string(index=False))
    
    # Create summary plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Create bar chart comparing systems
        plt.figure(figsize=(12, 8))
        systems = comparison_data["System"]
        x = np.arange(len(systems))
        width = 0.25
        
        # Plot bars
        plt.bar(x - width, comparison_data["Faithfulness"], width, label='Faithfulness')
        plt.bar(x, comparison_data["Correctness"], width, label='Correctness')
        plt.bar(x + width, comparison_data["Combined"], width, label='Combined')
        
        # Add labels and legend
        plt.xlabel('System')
        plt.ylabel('Score')
        plt.title('RAG System Comparison')
        plt.xticks(x, systems)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, "system_comparison.png"), dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {os.path.join(output_dir, 'system_comparison.png')}")
    except:
        print("Matplotlib not available - skipping plot generation")

def compare_systems(rag_output_file, baseline_file, output_file, api_key=None):
    """Compare a RAG system output with a baseline."""

    # If output_file is a directory, create a timestamped file
    if os.path.isdir(output_file) or not output_file.endswith('.json'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.basename(rag_output_file).replace('.json', '')
        output_file = os.path.join(output_file, f"eval_{output_base}_{timestamp}.json")
        print(f"Results will be saved to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load data
    with open(rag_output_file, 'r') as f:
        rag_data = json.load(f)
    
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    # Add system names
    rag_data["system_name"] = "Your RAG System"
    baseline_data["system_name"] = "Baseline"
    
    # Initialize evaluator
    evaluator = RAGEvaluator(api_key=api_key)
    
    # Evaluate both systems
    print("Evaluating your RAG system...")
    rag_results = evaluator.evaluate_rag_output(rag_data, baseline_data)
    
    print("Evaluating baseline system...")
    baseline_results = evaluator.evaluate_rag_output(baseline_data, baseline_data)
    
    # Combine results
    comparison = {
        "question": rag_data["question"],
        "rag_system": {
            "faithfulness_score": rag_results["faithfulness_score"],
            "correctness_score": rag_results["correctness_score"],
            "combined_score": rag_results["combined_score"],
            "passages_count": rag_results["passages_count"]
        },
        "baseline": {
            "faithfulness_score": baseline_results["faithfulness_score"],
            "correctness_score": baseline_results["correctness_score"],
            "combined_score": baseline_results["combined_score"],
            "passages_count": baseline_results["passages_count"]
        },
        "detailed_evaluations": {
            "rag_system": {
                "faithfulness_evaluation": rag_results["faithfulness_evaluation"],
                "correctness_evaluation": rag_results.get("correctness_evaluation")
            },
            "baseline": {
                "faithfulness_evaluation": baseline_results["faithfulness_evaluation"],
                "correctness_evaluation": baseline_results.get("correctness_evaluation")
            }
        }
    }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Question: {rag_data['question']}")
    print("\nYour RAG System:")
    print(f"  Faithfulness: {rag_results['faithfulness_score']:.2f} (-1 to 1)")
    if rag_results['correctness_score'] is not None:
        print(f"  Correctness: {rag_results['correctness_score']:.2f} (-1 to 2)")
        print(f"  Combined: {rag_results['combined_score']:.2f}")
    print(f"  Passages: {rag_results['passages_count']}")
    
    print("\nBaseline:")
    print(f"  Faithfulness: {baseline_results['faithfulness_score']:.2f} (-1 to 1)")
    if baseline_results['correctness_score'] is not None:
        print(f"  Correctness: {baseline_results['correctness_score']:.2f} (-1 to 2)")
        print(f"  Combined: {baseline_results['combined_score']:.2f}")
    print(f"  Passages: {baseline_results['passages_count']}")
    
    return comparison

def batch_evaluate(rag_dir, baseline_dir, output_dir, api_key=None):
    """Evaluate multiple RAG outputs against their corresponding baselines."""
    os.makedirs(output_dir, exist_ok=True)
    # Create timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a timestamped subdirectory for this run
    run_dir = os.path.join(output_dir, f"eval_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")

    # Find matching files
    rag_files = [f for f in os.listdir(rag_dir) if f.endswith('.json')]
    
    results = []
    
    for rag_file in tqdm(rag_files, desc="Evaluating questions"):
        baseline_file = rag_file  # Assumes matching names
        
        rag_path = os.path.join(rag_dir, rag_file)
        baseline_path = os.path.join(baseline_dir, baseline_file)
        
        if not os.path.exists(baseline_path):
            print(f"Warning: No matching baseline for {rag_file}. Skipping.")
            continue
        
        # Define output path
        output_file = os.path.join(run_dir, f"eval_{rag_file}")
        
        # Run comparison
        try:
            comparison = compare_systems(rag_path, baseline_path, output_file, api_key)
            results.append(comparison)
        except Exception as e:
            print(f"Error evaluating {rag_file}: {e}")
    
    # Generate summary statistics
    if results:
        generate_summary(results, os.path.join(run_dir, "summary.json"))
        generate_summary_tables(results, run_dir)
    
    return results

def generate_summary(results, output_file):
    """Generate summary statistics from evaluation results."""
    summary = {
        "total_questions": len(results),
        "rag_system": {
            "faithfulness": {
                "mean": np.mean([r["rag_system"]["faithfulness_score"] for r in results]),
                "std": np.std([r["rag_system"]["faithfulness_score"] for r in results]),
                "min": min([r["rag_system"]["faithfulness_score"] for r in results]),
                "max": max([r["rag_system"]["faithfulness_score"] for r in results])
            },
            "correctness": {
                "mean": np.mean([r["rag_system"]["correctness_score"] for r in results if r["rag_system"]["correctness_score"] is not None]),
                "std": np.std([r["rag_system"]["correctness_score"] for r in results if r["rag_system"]["correctness_score"] is not None]),
                "min": min([r["rag_system"]["correctness_score"] for r in results if r["rag_system"]["correctness_score"] is not None]),
                "max": max([r["rag_system"]["correctness_score"] for r in results if r["rag_system"]["correctness_score"] is not None])
            },
            "combined": {
                "mean": np.mean([r["rag_system"]["combined_score"] for r in results if r["rag_system"]["combined_score"] is not None]),
                "std": np.std([r["rag_system"]["combined_score"] for r in results if r["rag_system"]["combined_score"] is not None]),
                "min": min([r["rag_system"]["combined_score"] for r in results if r["rag_system"]["combined_score"] is not None]),
                "max": max([r["rag_system"]["combined_score"] for r in results if r["rag_system"]["combined_score"] is not None])
            }
        },
        "baseline": {
            "faithfulness": {
                "mean": np.mean([r["baseline"]["faithfulness_score"] for r in results]),
                "std": np.std([r["baseline"]["faithfulness_score"] for r in results]),
                "min": min([r["baseline"]["faithfulness_score"] for r in results]),
                "max": max([r["baseline"]["faithfulness_score"] for r in results])
            },
            "correctness": {
                "mean": np.mean([r["baseline"]["correctness_score"] for r in results if r["baseline"]["correctness_score"] is not None]),
                "std": np.std([r["baseline"]["correctness_score"] for r in results if r["baseline"]["correctness_score"] is not None]),
                "min": min([r["baseline"]["correctness_score"] for r in results if r["baseline"]["correctness_score"] is not None]),
                "max": max([r["baseline"]["correctness_score"] for r in results if r["baseline"]["correctness_score"] is not None])
            },
            "combined": {
                "mean": np.mean([r["baseline"]["combined_score"] for r in results if r["baseline"]["combined_score"] is not None]),
                "std": np.std([r["baseline"]["combined_score"] for r in results if r["baseline"]["combined_score"] is not None]),
                "min": min([r["baseline"]["combined_score"] for r in results if r["baseline"]["combined_score"] is not None]),
                "max": max([r["baseline"]["combined_score"] for r in results if r["baseline"]["combined_score"] is not None])
            }
        },
        "comparison": {
            "faithfulness_diff": np.mean([r["rag_system"]["faithfulness_score"] - r["baseline"]["faithfulness_score"] for r in results]),
            "correctness_diff": np.mean([r["rag_system"]["correctness_score"] - r["baseline"]["correctness_score"] for r in results if r["rag_system"]["correctness_score"] is not None and r["baseline"]["correctness_score"] is not None]),
            "combined_diff": np.mean([r["rag_system"]["combined_score"] - r["baseline"]["combined_score"] for r in results if r["rag_system"]["combined_score"] is not None and r["baseline"]["combined_score"] is not None]),
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary to console
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total questions evaluated: {summary['total_questions']}")
    
    print("\nYour RAG System (average scores):")
    print(f"  Faithfulness: {summary['rag_system']['faithfulness']['mean']:.2f} ± {summary['rag_system']['faithfulness']['std']:.2f} (-1 to 1)")
    print(f"  Correctness: {summary['rag_system']['correctness']['mean']:.2f} ± {summary['rag_system']['correctness']['std']:.2f} (-1 to 2)")
    print(f"  Combined: {summary['rag_system']['combined']['mean']:.2f} ± {summary['rag_system']['combined']['std']:.2f}")
    
    print("\nBaseline (average scores):")
    print(f"  Faithfulness: {summary['baseline']['faithfulness']['mean']:.2f} ± {summary['baseline']['faithfulness']['std']:.2f} (-1 to 1)")
    print(f"  Correctness: {summary['baseline']['correctness']['mean']:.2f} ± {summary['baseline']['correctness']['std']:.2f} (-1 to 2)")
    print(f"  Combined: {summary['baseline']['combined']['mean']:.2f} ± {summary['baseline']['combined']['std']:.2f}")
    
    print("\nComparison (RAG System - Baseline):")
    print(f"  Faithfulness difference: {summary['comparison']['faithfulness_diff']:.2f}")
    print(f"  Correctness difference: {summary['comparison']['correctness_diff']:.2f}")
    print(f"  Combined difference: {summary['comparison']['combined_diff']:.2f}")
    
    return summary

def generate_summary_tables(results, output_dir):
    """Generate summary tables in CSV format."""
    # Create detailed results table
    details = []
    for r in results:
        details.append({
            "question": r["question"],
            "rag_faithfulness": r["rag_system"]["faithfulness_score"],
            "rag_correctness": r["rag_system"]["correctness_score"],
            "rag_combined": r["rag_system"]["combined_score"],
            "baseline_faithfulness": r["baseline"]["faithfulness_score"],
            "baseline_correctness": r["baseline"]["correctness_score"],
            "baseline_combined": r["baseline"]["combined_score"],
            "faithfulness_diff": r["rag_system"]["faithfulness_score"] - r["baseline"]["faithfulness_score"],
            "correctness_diff": r["rag_system"]["correctness_score"] - r["baseline"]["correctness_score"] if r["rag_system"]["correctness_score"] is not None and r["baseline"]["correctness_score"] is not None else None,
            "combined_diff": r["rag_system"]["combined_score"] - r["baseline"]["combined_score"] if r["rag_system"]["combined_score"] is not None and r["baseline"]["combined_score"] is not None else None
        })
    
    details_df = pd.DataFrame(details)
    details_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    
    # Create system comparison table
    system_comparison = pd.DataFrame({
        "System": ["Your RAG System", "Baseline"],
        "Avg. Faithfulness": [
            details_df["rag_faithfulness"].mean(),
            details_df["baseline_faithfulness"].mean()
        ],
        "Avg. Correctness": [
            details_df["rag_correctness"].mean(),
            details_df["baseline_correctness"].mean()
        ],
        "Avg. Combined": [
            details_df["rag_combined"].mean(),
            details_df["baseline_combined"].mean()
        ]
    })
    
    system_comparison.to_csv(os.path.join(output_dir, "system_comparison.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare multiple RAG systems")
    parser.add_argument("--rag_files", nargs='+', required=True, help="Paths to RAG system JSONL files")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSONL file")
    parser.add_argument("--output", default="evaluation_results", help="Directory to save results")
    parser.add_argument("--api_key", help="API key for SCADS.AI (optional if stored in ~/.scadsai-api-key)")
    parser.add_argument("--sample", type=int, help="Evaluate only a sample of questions (specify number)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Evaluate systems
    evaluate_multiple_systems(args.rag_files, args.baseline, args.output, args.api_key)

if __name__ == "__main__":
    main()