import torch
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel

class PineconeRetriever:
    """
    Retriever that uses Pinecone to access the FineWeb corpus.
    
    This retriever implements the same interface as other retrievers in MAIN-RAG
    but uses Pinecone for vector search instead of a local database.
    """
    
    def __init__(self, 
                 api_key,
                 index_name="fineweb10bt-512-0w-e5-base-v2",
                 namespace="default",
                 embedding_model="intfloat/e5-base-v2"):
        """
        Initialize the Pinecone retriever.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
            embedding_model: Model to use for embedding queries
        """
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize Pinecone client
        print(f"Initializing Pinecone with index: {index_name}")
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(name=self.index_name)
        
        # Load embedding model and tokenizer
        print(f"Loading embedding model: {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"Embedding model loaded on {self.device}")
    
    def _average_pool(self, last_hidden_states, attention_mask):
        """
        Perform average pooling on model outputs.
        
        Args:
            last_hidden_states: Output from the model
            attention_mask: Attention mask to ignore padding
            
        Returns:
            Pooled embeddings
        """
        # Mask out padding tokens
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        # Average over sequence length
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def _embed_query(self, query):
        """
        Convert a query to an embedding vector.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector as a list of floats
        """
        # Format query with prefix as required by e5 model
        query_text = f"query: {query}"
        
        # Tokenize
        inputs = self.tokenizer(query_text, 
                               padding=True, 
                               truncation=True, 
                               return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._average_pool(outputs.last_hidden_state, inputs["attention_mask"])
            # Normalize to unit length (required for cosine similarity)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Return as list for Pinecone
        return embeddings[0].cpu().numpy().tolist()
    
    def retrieve(self, query, top_k=20):
        """
        Retrieve documents from Pinecone for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of document texts
        """
        print(f"Retrieving documents for query: {query}")
        
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
        
        print(f"Retrieved {len(documents)} documents")
        return documents
