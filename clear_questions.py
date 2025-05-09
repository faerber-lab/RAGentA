import json

# Input and output file paths
input_file = "500-questions.jsonl"  # Replace with your actual file path
output_file = "cleaned_500_questions.jsonl"

# Process the file
with open(input_file, "r", encoding="utf-8") as infile, open(
    output_file, "w", encoding="utf-8"
) as outfile:
    for i, line in enumerate(infile):
        try:
            # Parse the JSON object from each line
            record = json.loads(line.strip())

            # Extract the question
            if "question" in record:
                # Create the simplified output format
                output_record = {"id": i, "question": record["question"]}

                # Write to the output file with ensure_ascii=False to preserve Unicode characters
                outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            print(f"Error parsing line {i+1}. Skipping.")
        except Exception as e:
            print(f"Error processing line {i+1}: {str(e)}. Skipping.")

print(f"Extraction complete. Questions saved to {output_file}")
