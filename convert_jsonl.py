import json

# Read JSONL file
questions = []
with open("datamorgana_questions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

# Write JSON file
with open("datamorgana_questions.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2)
