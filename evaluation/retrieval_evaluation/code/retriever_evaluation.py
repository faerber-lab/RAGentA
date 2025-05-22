def evaluate_single_rag_mrr(rag_outputs, golden_ids, k=5):
    for idx, doc_id in enumerate(rag_outputs[:k]):
        if doc_id in golden_ids:
            return 1 / (idx + 1)
    return 0.0


def evaluate_corpus_rag_mrr(rag_outputs_list, golden_ids_list, k=5):
    scores = [
        evaluate_single_rag_mrr(rag_outputs, golden_ids, k)
        for rag_outputs, golden_ids in zip(rag_outputs_list, golden_ids_list)
    ]
    return sum(scores) / len(scores) if scores else 0.0
    
def evaluate_single_rag_recall(rag_outputs, golden_ids, k=20):
    """
    Evaluate Recall@k for a single query.

    Args:
        rag_outputs (List[str]): Retrieved document IDs ranked by relevance.
        golden_ids (List[str]): List of correct document IDs for the query.
        k (int): Number of top documents to consider (default 20).

    Returns:
        float: Recall@k score for the single query.
    """
    retrieved_k = set(rag_outputs[:k])
    relevant = set(golden_ids)
    hits = retrieved_k.intersection(relevant)
    if not relevant:
        return 0.0
    return len(hits) / len(relevant)


def evaluate_single_rag_precision(rag_outputs, golden_ids, k=20):
    """
    Evaluate Precision@k for a single query.

    Args:
        rag_outputs (List[str]): Retrieved document IDs ranked by relevance.
        golden_ids (List[str]): List of correct document IDs for the query.
        k (int): Number of top documents to consider (default 20).

    Returns:
        float: Precision@k score for the single query.
    """
    retrieved_k = set(rag_outputs[:k])
    relevant = set(golden_ids)
    hits = retrieved_k.intersection(relevant)
    return len(hits) / k


def evaluate_single_rag_f1(rag_outputs, golden_ids, k=20):
    """
    Evaluate F1@k for a single query.

    Args:
        rag_outputs (List[str]): Retrieved document IDs ranked by relevance.
        golden_ids (List[str]): List of correct document IDs for the query.
        k (int): Number of top documents to consider (default 20).

    Returns:
        float: F1@k score for the single query.
    """
    precision = evaluate_single_rag_precision(rag_outputs, golden_ids, k)
    recall = evaluate_single_rag_recall(rag_outputs, golden_ids, k)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
  
def evaluate_corpus_rag_mrr(rag_outputs_list, golden_ids_list, k=5):
    """
    Compute average MRR@k over a corpus.

    Args:
        rag_outputs_list (List[List[str]]): List of retrieved document ID lists for each query.
        golden_ids_list (List[List[str]]): List of golden document ID lists for each query.
        k (int): Number of top documents to consider.

    Returns:
        float: Average MRR@k score.
    """
    scores = [
        evaluate_single_rag_mrr(rag_outputs, golden_ids, k)
        for rag_outputs, golden_ids in zip(rag_outputs_list, golden_ids_list)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_corpus_rag_recall(rag_outputs_list, golden_ids_list, k=20):
    """
    Compute average Recall@k over a corpus.

    Returns:
        float: Average Recall@k score.
    """
    scores = [
        evaluate_single_rag_recall(rag_outputs, golden_ids, k)
        for rag_outputs, golden_ids in zip(rag_outputs_list, golden_ids_list)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_corpus_rag_precision(rag_outputs_list, golden_ids_list, k=20):
    """
    Compute average Precision@k over a corpus.

    Returns:
        float: Average Precision@k score.
    """
    scores = [
        evaluate_single_rag_precision(rag_outputs, golden_ids, k)
        for rag_outputs, golden_ids in zip(rag_outputs_list, golden_ids_list)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_corpus_rag_f1(rag_outputs_list, golden_ids_list, k=20):
    """
    Compute average F1@k over a corpus.

    Returns:
        float: Average F1@k score.
    """
    scores = [
        evaluate_single_rag_f1(rag_outputs, golden_ids, k)
        for rag_outputs, golden_ids in zip(rag_outputs_list, golden_ids_list)
    ]
    return sum(scores) / len(scores) if scores else 0.0
