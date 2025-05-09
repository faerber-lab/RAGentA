#!/usr/bin/env python3
import json
import os
import argparse
import math

def split_jsonl(input_file, output_dir, num_splits):
    """Split a JSONL file into multiple smaller files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read all questions
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = [line for line in f if line.strip()]
    
    # Calculate questions per split
    total_questions = len(questions)
    questions_per_split = math.ceil(total_questions / num_splits)
    
    print(f"Found {total_questions} questions, creating {num_splits} splits with ~{questions_per_split} questions each")
    
    # Create the split files
    for i in range(num_splits):
        start_idx = i * questions_per_split
        end_idx = min((i + 1) * questions_per_split, total_questions)
        
        if start_idx >= total_questions:
            break
            
        output_file = os.path.join(output_dir, f"questions_split_{i+1}.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for j in range(start_idx, end_idx):
                f.write(questions[j])
        
        print(f"Created {output_file} with questions {start_idx+1} to {end_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a JSONL file into multiple parts")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_dir", type=str, default="split_questions", help="Output directory")
    parser.add_argument("--num_splits", type=int, default=5, help="Number of splits to create")
    
    args = parser.parse_args()
    split_jsonl(args.input_file, args.output_dir, args.num_splits)
