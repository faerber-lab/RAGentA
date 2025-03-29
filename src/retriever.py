from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json
from datasets import load_dataset
from tqdm import tqdm

class WikipediaRetriever:
    def __init__(self, model_name='facebook/contriever-msmarco', wiki_path='data/documents/wikipedia', use_cached=True):
        """
        Wikipedia-based document retriever.
        
        Args:
            model_name: SentenceTransformer model to use for embeddings
            wiki_path: Path to store Wikipedia data
            use_cached: Whether to use cached Wikipedia embeddings if available
        """
        self.model = SentenceTransformer(model_name)
        self.wiki_path = wiki_path
        os.makedirs(wiki_path, exist_ok=True)
        
        # Check if we have cached Wikipedia data
        self.index_path = os.path.join(wiki_path, "faiss_index.bin")
        self.passages_path = os.path.join(wiki_path, "passages.json")
        
        if use_cached and os.path.exists(self.index_path) and os.path.exists(self.passages_path):
            print("Loading cached Wikipedia index and passages...")
            self.index = faiss.read_index(self.index_path)
            with open(self.passages_path, 'r', encoding='utf-8') as f:
                self.passages = json.load(f)
            print(f"Loaded {len(self.passages)} passages")
        else:
            print("Preparing Wikipedia data...")
            self.prepare_wikipedia()
    
    def prepare_wikipedia(self, max_passages=100000):  # Limit for testing
        """Prepare Wikipedia data for retrieval."""
        print("Loading Wikipedia dataset...")
        # Use a smaller subset for testing (for full reproduction, use the complete dataset)
        wiki_data = load_dataset("wikipedia", "20220301.en", split="train[:100000]", trust_remote_code=True)
        
        # Process Wikipedia articles into passages
        print("Processing Wikipedia articles into passages...")
        self.passages = []
        
        for article in tqdm(wiki_data):
            title = article["title"]
            text = article["text"]
            
            # Simple passage chunking (500 character chunks with 100 character overlap)
            # For better results, use a more sophisticated chunking strategy
            chunk_size = 500
            overlap = 100
            
            if len(text) <= chunk_size:
                self.passages.append(f"{title}: {text}")
            else:
                for i in range(0, len(text) - overlap, chunk_size - overlap):
                    chunk = text[i:i + chunk_size]
                    self.passages.append(f"{title}: {chunk}")
            
            if len(self.passages) >= max_passages:
                break
        
        # Create embeddings
        print(f"Creating embeddings for {len(self.passages)} passages...")
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(self.passages), batch_size)):
            batch = self.passages[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        
        # Create FAISS index
        print("Creating FAISS index...")
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Save data for future use
        print("Saving index and passages...")
        faiss.write_index(self.index, self.index_path)
        with open(self.passages_path, 'w', encoding='utf-8') as f:
            json.dump(self.passages, f)
    
    def retrieve(self, query, top_k=20):
        """Retrieve top-k relevant documents for a query."""
        query_embedding = self.model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            min(top_k, len(self.passages))
        )
        
        # Return the documents
        return [self.passages[i] for i in indices[0]]
