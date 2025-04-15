# pinecone_retriever.py

import torch
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel

class PineconeRetriever:
    def __init__(self, api_key, index_name="fineweb10bt-512-0w-e5-base-v2", namespace="default"):
        """Retriever that uses Pinecone to access the FineWeb corpus."""
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(name=self.index_name)
        
        # Load embedding model
        self.model_name = "intfloat/e5-base-v2"  # Same as used in FineWeb index
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
    def _embed_query(self, query):
        """Convert query to embedding vector."""
        # Format query with prefix as required by e5 model
        query_text = f"query: {query}"
        
        # Tokenize and get model outputs
        inputs = self.tokenizer(query_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Average pooling and normalization
        attention_mask = inputs["attention_mask"]
        last_hidden = outputs.last_hidden_state
        masked_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        normalized = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Convert to list for Pinecone
        return normalized[0].cpu().numpy().tolist()
    
    def retrieve(self, query, top_k=20):
        """Retrieve documents from Pinecone for a given query."""
        # Create query embedding
        query_embedding = self._embed_query(query)
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_values=False,
            namespace=self.namespace,
            include_metadata=True
        )
        
        # Extract document texts
        documents = []
        for match in results["matches"]:
            documents.append(match["metadata"]["text"])
        
        return documents
