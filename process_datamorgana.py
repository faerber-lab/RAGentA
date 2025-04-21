# process_datamorgana.py
import json
import requests
import os

def process_datamorgana_results(request_id, api_key, output_file="datamorgana_questions.json"):
    """Process the results from DataMorgana bulk generation."""
    # Fetch the results
    response = requests.get(
        "https://api.ai71.ai/v1/fetch_generation_results",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"request_id": request_id}
    )
    
    if response.status_code != 200:
        print(f"Error fetching results: {response.status_code}")
        print(response.text)
        return
        
    if response.json()["status"] != "completed":
        print("Generation not completed yet")
        return
    
    # Get the file URL
    file_url = response.json()["file"]
    file_response = requests.get(file_url)
    
    # Parse the JSONL file
    qa_pairs = [json.loads(line) for line in file_response.text.splitlines()]
    
    # Convert to the format we need
    formatted_data = []
    for pair in qa_pairs:
        formatted_data.append({
            "question": pair["question"],
            "answer": pair["answer"],
            "context": pair["context"]
        })
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Processed {len(formatted_data)} questions and saved to {output_file}")
    return formatted_data

# If you already have your Request ID
if __name__ == "__main__":
    api_key = "YOUR_API_KEY"
    request_id = "YOUR_REQUEST_ID" 
    process_datamorgana_results(request_id, api_key)
