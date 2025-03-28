from datasets import load_dataset
import json
import os

os.makedirs("data/benchmarks/triviaqa", exist_ok=True)

dataset = load_dataset("trivia_qa", "unfiltered", split="validation")

formatted_data = []
for item in dataset:
    formatted_data.append({
        "question": item["question"],
        "answer": item["answer"]["value"],
        "aliases": item["answer"]["aliases"]
    })

with open("data/benchmarks/triviaqa/validation.json", "w") as f:
    json.dump(formatted_data, f, indent=2)

print(f"Saved {len(formatted_data)} TriviaQA validation examples")
