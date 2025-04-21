import torch
import boto3
from pinecone import Pinecone
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
from transformers import AutoModel, AutoTokenizer
from functools import cache

# AWS configuration
AWS_PROFILE_NAME = "sigir-participant"
AWS_REGION_NAME = "us-east-1"

# Pinecone configuration
PINECONE_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
PINECONE_NAMESPACE = "default"

# OpenSearch configuration
OPENSEARCH_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"


class HybridRetriever:
    def __init__(self, alpha=0.7, top_k=20):
        """
        Initialize a hybrid retriever that combines dense and sparse retrieval.

        Args:
            alpha: Weight for semantic search (0-1). Higher means more weight to semantic search.
            top_k: Number of documents to retrieve
        """
        self.alpha = alpha
        self.top_k = top_k

        # Initialize embedding model
        print("Loading embedding model...")
        self.model_name = "intfloat/e5-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # Initialize Pinecone
        print("Connecting to Pinecone...")
        self.pinecone_api_key = self._get_ssm_secret("/pinecone/ro_token")
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.pinecone_index = self.pc.Index(name=PINECONE_INDEX_NAME)

        # Initialize OpenSearch
        print("Connecting to OpenSearch...")
        self.opensearch_client = self._get_opensearch_client()

        print("Hybrid retriever initialized successfully")

    def _get_ssm_secret(self, key, profile=None, region=AWS_REGION_NAME):
        """Get a secret from AWS SSM."""
        # Use environment variables for AWS credentials
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region,
        )
        ssm = session.client("ssm")
        return ssm.get_parameter(Name=key, WithDecryption=True)["Parameter"]["Value"]

    def _get_opensearch_client(self, profile=AWS_PROFILE_NAME, region=AWS_REGION_NAME):
        """Get an OpenSearch client."""
        credentials = boto3.Session(profile_name=profile).get_credentials()
        auth = AWSV4SignerAuth(credentials, region=region)
        host_name = self._get_ssm_value(
            "/opensearch/endpoint", profile=profile, region=region
        )

        return OpenSearch(
            hosts=[{"host": host_name, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

    def _get_ssm_value(self, key, profile=AWS_PROFILE_NAME, region=AWS_REGION_NAME):
        """Get a cleartext value from AWS SSM."""
        session = boto3.Session(profile_name=profile, region_name=region)
        ssm = session.client("ssm")
        return ssm.get_parameter(Name=key)["Parameter"]["Value"]

    def _embed_query(self, query):
        """Create embeddings for a query."""
        query_with_prefix = f"query: {query}"

        with torch.no_grad():
            inputs = self.tokenizer(
                [query_with_prefix], return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            # Use average pooling
            attention_mask = inputs["attention_mask"]
            last_hidden = outputs.last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().tolist()

    def _normalize_scores(self, results, score_key):
        """Normalize scores to range 0-1."""
        if not results:
            return []

        scores = [res[score_key] for res in results]
        min_score, max_score = min(scores), max(scores)

        for res in results:
            if max_score > min_score:
                res["normalized_score"] = (res[score_key] - min_score) / (
                    max_score - min_score
                )
            else:
                res["normalized_score"] = 0.0

        return results

    def _parse_pinecone_results(self, results):
        """Parse Pinecone results into a standard format."""
        parsed = []
        for match in results.matches:
            parsed.append(
                {
                    "id": match.id,
                    "text": match.metadata.get("text", ""),
                    "score": match.score,
                }
            )
        return parsed

    def _parse_opensearch_results(self, results):
        """Parse OpenSearch results into a standard format."""
        hits = results.get("hits", {}).get("hits", [])
        parsed = []
        for hit in hits:
            parsed.append(
                {
                    "id": hit["_id"],
                    "text": hit["_source"].get("text", ""),
                    "score": hit["_score"],
                }
            )
        return parsed

    def retrieve(self, query, top_k=None):
        """
        Retrieve documents using hybrid search (semantic + keyword).

        Args:
            query: The search query
            top_k: Number of documents to retrieve

        Returns:
            List of documents
        """
        if top_k is None:
            top_k = self.top_k

        expanded_top_k = min(top_k * 3, 1000)  # Retrieve more docs to rerank

        # Semantic search (Pinecone)
        print(f"Performing semantic search for: {query}")
        query_embedding = self._embed_query(query)
        pinecone_results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=expanded_top_k,
            include_values=False,
            namespace=PINECONE_NAMESPACE,
            include_metadata=True,
        )

        # Keyword search (OpenSearch)
        print("Performing keyword search")
        opensearch_results = self.opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body={"query": {"match": {"text": query}}, "size": expanded_top_k},
        )

        # Parse results
        semantic_results = self._parse_pinecone_results(pinecone_results)
        keyword_results = self._parse_opensearch_results(opensearch_results)

        # Normalize scores
        semantic_results = self._normalize_scores(semantic_results, "score")
        keyword_results = self._normalize_scores(keyword_results, "score")

        # Merge results
        combined = {}
        for res in semantic_results:
            combined[res["id"]] = {
                "id": res["id"],
                "text": res["text"],
                "semantic_score": res["normalized_score"],
                "keyword_score": 0.0,
            }

        for res in keyword_results:
            if res["id"] in combined:
                combined[res["id"]]["keyword_score"] = res["normalized_score"]
            else:
                combined[res["id"]] = {
                    "id": res["id"],
                    "text": res["text"],
                    "semantic_score": 0.0,
                    "keyword_score": res["normalized_score"],
                }

        # Calculate hybrid scores
        for res in combined.values():
            res["final_score"] = (
                self.alpha * res["semantic_score"]
                + (1 - self.alpha) * res["keyword_score"]
            )

        # Sort and limit
        ranked_results = sorted(
            combined.values(), key=lambda x: x["final_score"], reverse=True
        )[:top_k]

        # Extract just the text
        documents = [res["text"] for res in ranked_results]
        print(f"Retrieved {len(documents)} documents using hybrid search")

        return documents
