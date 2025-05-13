# RAGent - Retrieval-Augmented Generation Agent Framwork
RAGent is an advanced framework for Retrieval-Augmented Generation that improves answer generation through a multi-agent architecture with citation tracking, claim analysis, and follow-up question processing. 

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
RAGent uses AWS services for document retrieval. You'll need to set up AWS credentials:
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

## Running RAGent
RAGent can be run on a single question or a batch of questions from a JSON/JSONL file.
### Process a Single Question
```bash
python run_RAGent.py --model tiiuae/falcon-3-10b-instruct --n 0.5 --alpha 0.7 --top_k 20 --single_question "Your question here?"
```
### Process Questions from a Dataset
```bash
python run_RAGent.py --model tiiuae/falcon-3-10b-instruct --n 0.5 --alpha 0.7 --top_k 20 --data_file your_questions.jsonl --output_format jsonl
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

## Architecture Details
RAGent uses a four-agent architecture:
- **Agent 1 (Predictor)**: Generates candidate answers for each retrieved document
- **Agent 2 (Judge)**: Evaluates document relevance for the query
- **Agent 3 (Final-Predictor)**: Generates the final answer with citations
- **Agent 4 (Claim Judge)**: Analyzes claims in the answer and identifies knowledge gaps

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

## Citation
If you use RAGent in your research, please cite:
```
@software{RAGent2025,
  author = {Schreieder, Tobias and Besrour, Ines and He, Jingbo},
  title = {RAGent: Retrieval-Augmented Generation Agent Framwork},
  year = {2025},
  publisher = {GitHub},
  url = {git@github.com:tobiasschreieder/LiveRAG.git}
}
```
