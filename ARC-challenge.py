from datasets import load_dataset
import json
import os

os.makedirs("data/benchmarks/arc_challenge", exist_ok=True)

dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")

formatted_data = []
for item in dataset:
    formatted_data.append({
        "question": item["question"],
        "choices": item["choices"]["text"],
        "answer_idx": item["choices"]["label"].index(item["answerKey"]),
        "answer": item["choices"]["text"][item["choices"]["label"].index(item["answerKey"])]
    })

with open("data/benchmarks/arc_challenge/test.json", "w") as f:
    json.dump(formatted_data, f, indent=2)

print(f"Saved {len(formatted_data)} ARC-Challenge test examples")
