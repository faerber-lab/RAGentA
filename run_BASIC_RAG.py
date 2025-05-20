import argparse
import json
import time
import datetime
import os
import logging
import numpy as np
from tqdm import tqdm
import re
import random
import string

# Generate a unique ID for log filename
def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/basic_rag_runner_{timestamp}_{random_str}.log"

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging with unique filename
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("BASIC_RAG_Runner")

# Import your existing retriever
from hybrid_retriever import HybridRetriever

class BasicRAG:
    def __init__(self, retriever, agent_model=None, top_k=10, falcon_api_key=None):
        """
        Basic RAG implementation without multi-agent filtering.
        
        Args:
            retriever: Document retriever instance
            agent_model: Model name or pre-initialized agent
            top_k: Number of documents to retrieve and use
        """
        self.retriever = retriever
        self.top_k = top_k
        
        # Initialize LLM agent
        if isinstance(agent_model, str):
            if "falcon" in agent_model.lower() and falcon_api_key:
                # Initialize Falcon agent if using Falcon API
                from api_agent import FalconAgent
                self.agent = FalconAgent(api_key=falcon_api_key)
                logger.info(f"Using Falcon agent with API")
            else:
                # Initialize local LLM agent
                from local_agent import LLMAgent
                self.agent = LLMAgent(agent_model)
                logger.info(f"Using local LLM agent with model {agent_model}")
        else:
            # Use pre-initialized agent
            self.agent = agent_model
            logger.info("Using pre-initialized agent")
    
    def _create_rag_prompt(self, query, documents):
        """Create prompt for the LLM with retrieved documents."""
        # Format documents
        docs_text = "\n\n".join(
            [
                f"Document {i+1}: {doc_text}"
                for i, (doc_text, _) in enumerate(documents)
            ]
        )
        
        return f"""You are an accurate and helpful AI assistant. Answer the question based ONLY on the information provided in the documents below. If the documents don't contain the necessary information to answer the question, simply state that you don't have enough information.

Documents:
{docs_text}

Question: {query}

Answer:"""
    
    def answer_query(self, query):
        """Process a query using basic RAG approach."""
        logger.info(f"Processing query with basic RAG: {query}")
        
        # Step 1: Retrieve documents
        logger.info(f"Retrieving top-{self.top_k} documents...")
        retrieved_docs = self.retriever.retrieve(query, top_k=self.top_k)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Create prompt with documents
        prompt = self._create_rag_prompt(query, retrieved_docs)
        
        # Step 3: Generate answer directly
        logger.info("Generating answer...")
        answer = self.agent.generate(prompt)
        
        # Log the answer
        logger.info(f"Answer: {answer}")
        
        # Return answer and debug info
        debug_info = {
            "raw_answer": answer,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt
        }
        
        return answer, debug_info

def load_questions(file_path):
    """Load questions from JSON or JSONL file."""
    # Determine if file is JSON or JSONL based on extension
    is_jsonl = file_path.lower().endswith(".jsonl")
    
    try:
        questions = []
        
        # JSONL format
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
        
        # JSON format
        else:
            logger.info(f"Loading questions from JSON file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # If JSON is an array, use it directly
                if isinstance(data, list):
                    questions = data
                    
                    # Add indices as IDs if not present
                    for i, question in enumerate(questions):
                        if "id" not in question:
                            question["id"] = i + 1
                # If JSON is an object, look for questions field
                elif isinstance(data, dict):
                    if "questions" in data:
                        questions = data["questions"]
                    elif "question" in data:
                        questions = [data]
                    else:
                        # Treat as single question
                        questions = [data]
                        
        logger.info(f"Loaded {len(questions)} questions")
        return questions
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading questions: {e}")
        return []

def format_result(result):
    """Format a result to match the required schema."""
    # Format passages
    passages = []
    for doc_text, doc_id in result.get("retrieved_docs", []):
        passages.append({"passage": doc_text, "doc_IDs": [doc_id]})
    
    # Create formatted result
    formatted_result = {
        "id": result.get("id", 0),
        "question": result.get("question", ""),
        "passages": passages,
        "final_prompt": result.get("prompt", ""),
        "answer": result.get("model_answer", "")
    }
    
    return formatted_result

def write_results_to_jsonl(results, output_file):
    """Write results to JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            formatted_result = format_result(result)
            f.write(json.dumps(formatted_result, ensure_ascii=False) + "\n")
    logger.info(f"Results written to {output_file}")

def write_result_to_json(result, output_file):
    """Write a single result to JSON file."""
    formatted_result = format_result(result)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Result written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Basic RAG with Hybrid Retrieval")
    parser.add_argument(
        "--model",
        type=str,
        default="tiiuae/falcon-3-10b-instruct",
        help="Model for LLM agent"
    )
    parser.add_argument(
        "--falcon_api_key",
        type=str,
        default=None,
        help="API key for Falcon API (only needed if using Falcon API)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="Weight for semantic search (0-1)"
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of documents to retrieve and use"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="questions.jsonl",
        help="File containing questions (JSON or JSONL)"
    )
    parser.add_argument(
        "--single_question",
        type=str,
        default=None,
        help="Process a single question instead of the entire dataset"
    )
    parser.add_argument(
        "--output_format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format: 'json' for single file, 'jsonl' for line-delimited JSON"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    args = parser.parse_args()
    
    # Initialize retriever
    logger.info(f"Initializing hybrid retriever with alpha={args.alpha}...")
    retriever = HybridRetriever(alpha=args.alpha, top_k=args.top_k)
    
    # Initialize BasicRAG
    logger.info(f"Initializing basic RAG with top_k={args.top_k}...")
    rag = BasicRAG(retriever, agent_model=args.model, top_k=args.top_k)

    # Initialize BasicRAG with falcon_api_key parameter
    logger.info(f"Initializing basic RAG with top_k={args.top_k}...")
    rag = BasicRAG(retriever, agent_model=args.model, top_k=args.top_k, 
                 falcon_api_key=args.falcon_api_key)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process a single question if specified
    if args.single_question:
        logger.info(f"\nProcessing single question: {args.single_question}")
        start_time = time.time()
        
        try:
            # Process the query
            answer, debug_info = rag.answer_query(args.single_question)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Create result object
            result = {
                "id": "single_question",
                "question": args.single_question,
                "model_answer": answer,
                "process_time": process_time,
                "retrieved_docs": debug_info["retrieved_docs"],
                "prompt": debug_info["prompt"]
            }
            
            logger.info(f"Answer: {answer}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            
            # Save result
            output_file = os.path.join(
                args.output_dir, f"basic_rag_single_question_{timestamp}.json"
            )
            write_result_to_json(result, output_file)
            
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            
        return
    
    # Load questions
    questions = load_questions(args.data_file)
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
            answer, debug_info = rag.answer_query(item["question"])
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Save result
            result = {
                "id": question_id,
                "question": item["question"],
                "reference_answer": item.get("answer", ""),
                "model_answer": answer,
                "process_time": process_time,
                "retrieved_docs": debug_info["retrieved_docs"],
                "prompt": debug_info["prompt"]
            }
            results.append(result)
            
            logger.info(f"Answer: {answer}")
            logger.info(f"Processing time: {process_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}", exc_info=True)
    
    # Save all results
    if results:
        if args.output_format == "jsonl":
            output_file = os.path.join(args.output_dir, f"basic_rag_answers_{timestamp}.jsonl")
            write_results_to_jsonl(results, output_file)
        else:  # json
            # Save each result as a separate JSON file
            for result in results:
                question_id = result["id"]
                output_file = os.path.join(
                    args.output_dir, f"basic_rag_answer_{question_id}_{timestamp}.json"
                )
                write_result_to_json(result, output_file)
    
    logger.info(f"\nProcessed {len(results)} questions.")
    
    # Print summary statistics
    if results:
        avg_time = sum(r["process_time"] for r in results) / len(results)
        logger.info(f"Average processing time: {avg_time:.2f} seconds")

if __name__ == "__main__":
    main()