# RAGentA: Multi-Agent Retrieval-Augmented Generation for Attributed Question Answering
RAGentA, a multi-agent retrieval-augmented generation (RAG) framework for attributed question answering. With the goal of trustworthy answer generation, RAGentA focuses on optimizing answer correctness, defined by coverage and relevance to the question and faithfulness, which measures the extent to which answers are grounded in retrieved documents

## Features
- **Multi-Agent Architecture**: Uses multiple specialized agents for document retrieval, relevance judgment, answer generation, and claim analysis
- **Hybrid Retrieval**: Combines semantic (dense) and keyword (sparse) search for better document retrieval
- **Citation Tracking**: Automatically tracks citations in generated answers to ensure factual accuracy
- **Claim Analysis**: Analyzes individual claims in answers to ensure relevance and identify knowledge gaps
- **Follow-Up Processing**: Generates follow-up questions for unanswered aspects and integrates additional knowledge
- **Evaluation Metrics**: Includes standard RAG evaluation metrics like MRR, Recall, Precision, and F1

## Requirements
- Python 3.8+
- PyTorch 2.0.0+
- CUDA-compatible GPU (recommended)
- AWS account with access to OpenSearch and Pinecone (for hybrid retrieval)

## Installation
1. Clone the repository:
```bash
git clone git@github.com:tobiasschreieder/LiveRAG.git
cd LiveRAG
```
3. Create and activate a virtual environment:
```python
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```
3. Install dependencies:
```python
pip install -r requirements.txt
```

## Configuration
### AWS Configuration
RAGentA uses AWS services for document retrieval. You'll need to set up AWS credentials:
1. Create AWS credentials file:
```bash
mkdir -p ~/.aws
```

2. Add your credentials to ~/.aws/credentials
```
[sigir-participant]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

3. Add your region to ~/.aws/config
```
[profile sigir-participant]
region = us-east-1
output = json
```

## Environment Variables
Set the following environment variables:
```bash
export AWS_PROFILE=sigir-participant
export AWS_REGION=us-east-1
export HUGGING_FACE_HUB_TOKEN=your_hf_token  # If needed for accessing models
```

## Running RAGentA
RAGentA can be run on a single question or a batch of questions from a JSON/JSONL file.
### Process a Single Question
```bash
python run_RAGentA.py --model tiiuae/Falcon3-10B-Instruct --n 0.5 --alpha 0.7 --top_k 20 --single_question "Your question here?"
```
### Process Questions from a Dataset
```bash
python run_RAGentA.py --model tiiuae/Falcon3-10B-Instruct --n 0.5 --alpha 0.7 --top_k 20 --data_file your_questions.jsonl --output_format jsonl
```
### Parameters
- `--model`: Model name or path (default: "tiiuae/falcon-3-10b-instruct")
- `--n`: Adjustment factor for adaptive judge bar (default: 0.5)
- `--alpha`: Weight for semantic search vs. keyword search (0-1, default: 0.7)
- `--top_k`: Number of documents to retrieve (default: 20)
- `--data_file`: File containing questions in JSON or JSONL format
- `--single_question`: Process a single question instead of a dataset
- `--output_format`: Output format - json, jsonl, or debug (default: jsonl)
- `--output_dir`: Directory to save results (default: "results")

## Input Format
The input file should be in JSON or JSONL format with the following structure:
```json
{
  "id": "question_id",
  "question": "The question text?"
}
```
Or for JSONL, each line should be:
```{"id": "question_id", "question": "The question text?"}```

### Output Format
Results are saved in the specified output format with the following structure:
```json
{
  "id": "question_id",
  "question": "The question text?",
  "passages": [
    {
      "passage": "Document content...",
      "doc_IDs": ["doc_id1", "doc_id2"]
    }
  ],
  "final_prompt": "Final prompt used for generation...",
  "answer": "Generated answer..."
}
```

## System Architecture
RAGentA uses a sophisticated multi-agent architecture to improve the quality of retrieval-augmented generation. Here's a detailed breakdown of how the system works:
- **Agent 1 (Predictor)**: Generates candidate answers for each retrieved document
- **Agent 2 (Judge)**: Evaluates document relevance for the query
- **Agent 3 (Final-Predictor)**: Generates the final answer with citations
- **Agent 4 (Claim Judge)**: Analyzes claims in the answer and identifies knowledge gaps
### Multi-Agent Architecture
The core of RAGentA is its 4-agent architecture that handles different aspects of the retrieval and generation process:
#### Agent 1: Predictor
- **Purpose**: Generates candidate answers for each retrieved document
- **Input**: Query + single document
- **Output**: Document-specific answer
- **Process**: For each retrieved document, Agent 1 creates a potential answer based solely on that document
- **Implementation**: Uses prompt template `_create_agent1_prompt()` in `RAGentA.py`
#### Agent 2: Judge
- **Purpose**: Evaluates document relevance and filters out noise
- **Input**: Query + document + candidate answer
- **Output**: Relevance score (log probability)
- **Process**: Calculates score as `log_probs["Yes"] - log_probs["No"]` for each document
- **Key Innovation**: Creates an adaptive threshold (τq) based on score distribution
- **Implementation**: Uses `get_log_probs()` method rather than regular generation
#### Agent 3: Final-Predictor
- **Purpose**: Generates comprehensive answer with citations
- **Input**: Query + filtered documents
- **Output**: Answer with proper [X] citations
- **Process**: Synthesizes information across filtered documents with strict citation requirements
- **Implementation**: Uses detailed prompt with citation guidelines in `_create_agent3_prompt()`
#### Agent 4: Claim Judge (EnhancedAgent4)
- **Purpose**: Analyzes answer quality and detects knowledge gaps
- **Input**: Query + answer with citations + claims + documents
- **Output**: Improved answer + follow-up questions if needed
- **Process**:
  1. Breaks answer into individual claims with citations
  2. Analyzes if the question has multiple components
  3. Determines which claims address which components
  4. Identifies any unanswered aspects
  5. Generates follow-up questions for missing information
  6. Retrieves new documents and integrates additional knowledge
- **Implementation**: Most complex agent, implemented as separate `EnhancedAgent4` class
### Hybrid Retrieval System
RAGentA uses a hybrid approach combining dense and sparse retrieval methods:
#### 1. Semantic Search (Dense Retrieval):
- Uses Pinecone vector database
- Embedding model: intfloat/e5-base-v2
- Provides context-aware document retrieval
#### 2. Keyword Search (Sparse Retrieval):
- Uses OpenSearch for traditional keyword matching
- Handles cases where semantic search may miss important keywords
#### 3. Hybrid Scoring:
- Combines scores with weighting parameter `alpha`
- Formula: `final_score = alpha * semantic_score + (1 - alpha) * keyword_score`
- Higher alpha (default 0.65) puts more emphasis on semantic search
### Question Analysis & Follow-up System
One of RAGentA's key innovations is how it analyzes questions and identifies when they're not fully answered:
1. **Question Structure Analysis**:
- Determines if question contains multiple distinct components
- Avoids artificially breaking a single question into parts
2. **Claim Mapping**:
- Maps each claim in the answer to specific question components
- Tracks which claims address which parts of the question
3. **Coverage Assessment**:
- Evaluates if each component is fully answered, partially answered, or not answered
- Uses regex patterns to extract coverage assessments from LLM output
4. **Follow-up Processing**:
- For unanswered components, generates standalone follow-up questions
- Retrieves new documents specifically for these follow-up questions
- Integrates new information into original answer
### Adaptive Judge Bar Mechanism
The system uses a statistical approach to determine document relevance:
1. Calculate mean score (τq) across all documents
2. Calculate standard deviation (σ) of scores
3. Set threshold as: `adjusted_tau_q = τq - n * σ` where n is a hyperparameter
4. Only documents with scores ≥ adjusted_tau_q are used
5. This adaptive threshold adjusts based on query difficulty and document quality
### Agent Implementations
RAGentA supports two types of agent implementations, but Local LLM Agent is strongly recommended:
1. **Local LLM Agent** (`LLMAgent` class):
- Runs models directly on local hardware (GPU/CPU)
- Supports various model precision formats (bfloat16, float16, float32)
- Optimized for different hardware configurations
- Uses Hugging Face transformers with device_map="auto"
2. **API-based Agent** (`FalconAgent` class):
- Connects to external API (AI71) for inference
- Uses exponential backoff for request retries
- Approximates log probabilities for Yes/No judgments
- Ideal when local hardware is insufficient for model size
### Information Flow
Here's how information flows through the system:
#### 1. Retrieval Phase:
- Query is sent to hybrid retriever
- Top-k documents retrieved (default 20)
- Documents combined from semantic and keyword search
#### 2. Initial Judgment Phase:
- Agent 1 generates answers for each document
- Agent 2 scores each document
- Adaptive threshold calculated
- Low-scoring documents filtered out
#### 3. Answer Generation Phase:
- Agent 3 generates comprehensive answer with citations
- Citations use [X] format where X is document number
- Claims extracted with citation mapping
#### 4. Analysis Phase:
- Agent 4 analyzes if question is completely answered
- Identifies any unanswered components
- Generates follow-up questions if needed
#### 5. Follow-up Phase (if needed):
- Retrieves new documents for follow-up questions
- Generates answers to follow-up questions
- Integrates new information with original answer

This multi-stage approach with specialized agents allows RAGentA to produce more accurate, comprehensive, and properly cited answers compared to simpler RAG approaches.

## Evaluation
To evaluate RAG performance, use the metrics in `RAG_evaluation.py`:
```python
from RAG_evaluation import evaluate_corpus_rag_mrr, evaluate_corpus_rag_recall

# Example usage
mrr_score = evaluate_corpus_rag_mrr(retrieved_docs_list, golden_docs_list, k=5)
recall_score = evaluate_corpus_rag_recall(retrieved_docs_list, golden_docs_list, k=20)
```

## License
This project is licensed under the BSD 2-Clause License - see the LICENSE file for details.

## Acknowledgments and Inspiration
RAGentA draws inspiration from the MAIN-RAG framework (Multi-Agent Filtering Retrieval-Augmented Generation) introduced by Chang et al. in their paper [MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](https://arxiv.org/abs/2501.00332). While RAGentA follows a similar multi-agent architecture approach for the first three agents, our implementation is independently developed and significantly extends the original concept through:
1. **Hybrid Retrieval System**: RAGentA implements an advanced hybrid retrieval approach that combines semantic (dense) and keyword (sparse) search with configurable weighting (α parameter) to improve document relevance
2. **Enhanced Agent-3**: Our implementation includes explicit citation tracking capabilities to improve answer transparency and traceability
3. **Additional Agent-4 (Claim Judge)**: RAGentA introduces a fourth agent that performs claim-by-claim analysis to identify gaps in knowledge and generate targeted follow-up questions
4. **Follow-up Processing**: RAGentA can retrieve additional information for unanswered aspects of questions through a novel follow-up question generation system

Please cite both the original MAIN-RAG paper and RAGentA in any work that uses this code:
```
@article{chang2025mainrag,
  title={MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation},
  author={Chang, Chia-Yuan and Jiang, Zhimeng and Rakesh, Vineeth and Pan, Menghai and Yeh, Chin-Chia Michael and Wang, Guanchu and Hu, Mingzhi and Xu, Zhichao and Zheng, Yan and Das, Mahashweta and Zou, Na},
  journal={arXiv preprint arXiv:2501.00332},
  year={2025}
}

@software{RAGentA2025,
  author = {Schreieder, Tobias and Besrour, Ines and He, Jingbo},
  title = {RAGentA: Multi-Agent Retrieval-Augmented Generation for Attributed Question Answering},
  year = {2025},
  publisher = {GitHub},
  url = {[git@github.com:tobiasschreieder/LiveRAG.git](https://github.com/tobiasschreieder/LiveRAG.git)}
}
```
