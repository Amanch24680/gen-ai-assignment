from typing import List


def recall_at_k(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """
    Calculate Recall@K:
    (number of relevant chunks retrieved in top K) / (total number of relevant chunks)
    Returns 0.0 if relevant_chunk_ids is empty.
    """
    if not relevant_chunk_ids or k <= 0:
        return 0.0

    top_k_retrieved = set(retrieved_chunk_ids[:k])
    relevant_set = set(relevant_chunk_ids)
    hits = len(top_k_retrieved & relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """
    Calculate Precision@K:
    (number of relevant chunks retrieved in top K) / K
    For cases where fewer than K results are returned, precision uses the number of returned results.
    Returns 0.0 if no results are retrieved or k <= 0.
    """
    if k <= 0:
        return 0.0

    top_k_retrieved = retrieved_chunk_ids[:k]
    if not top_k_retrieved:
        return 0.0

    effective_denominator = len(top_k_retrieved) if len(top_k_retrieved) < k else k
    hits = len(set(top_k_retrieved) & set(relevant_chunk_ids))
    return hits / effective_denominator


def reciprocal_rank(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str]) -> float:
    """
    Calculate Reciprocal Rank (RR):
    1 / (rank of the first relevant retrieved chunk)
    Returns 0.0 if no relevant chunk is found in retrieved results.
    """
    if not relevant_chunk_ids or not retrieved_chunk_ids:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    for rank_idx, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank_idx

    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: List[float]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR):
    mean(reciprocal_rank across queries)
    Returns 0.0 if list is empty.
    """
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
