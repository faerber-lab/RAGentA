from datasets import load_dataset
import json
import os

os.makedirs("data/documents", exist_ok=True)

# Load a small subset of Wikipedia for testing
wiki = load_dataset("wikipedia", "20220301.en", split="train[:10000]")

# Process and save articles
wiki_passages = []
for article in wiki:
    title = article["title"]
    text = article["text"]
    
    # Simple chunking (in practice, use more sophisticated chunking)
    chunk_size = 500
    if len(text) <= chunk_size:
        wiki_passages.append(f"{title}: {text}")
    else:
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            wiki_passages.append(f"{title}: {chunk}")

# Save to file
with open("data/documents/wikipedia_passages.json", "w") as f:
    json.dump(wiki_passages, f)

print(f"Saved {len(wiki_passages)} passages")
