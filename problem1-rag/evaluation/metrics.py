import math
import re
from collections import Counter
from typing import Any, List, Optional, Set

# Common English stopwords to ignore during lexical answer normalization
DEFAULT_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "this", "these", "those", "or", "which", "what", "where"
}


def normalize_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Normalize text string into a list of cleaned tokens.
    Steps: lowercase, strip punctuation, split whitespace, optionally filter stopwords.
    """
    if not text:
        return []
    clean_str = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t.strip() for t in clean_str.split() if t.strip()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS]
    return tokens


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


def hit_rate_at_k(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """
    Calculate Hit Rate@K (Binary Hit / Miss):
    1.0 if at least one relevant chunk is retrieved in top K, else 0.0.
    Returns 0.0 if relevant_chunk_ids is empty or k <= 0.
    """
    if not relevant_chunk_ids or k <= 0 or not retrieved_chunk_ids:
        return 0.0

    top_k_retrieved = set(retrieved_chunk_ids[:k])
    relevant_set = set(relevant_chunk_ids)
    return 1.0 if len(top_k_retrieved & relevant_set) > 0 else 0.0


def ndcg_at_k(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K (nDCG@K):
    DCG@K / IDCG@K
    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(K, |relevant|)} (1 / log2(i + 1))
    Returns 0.0 if relevant_chunk_ids is empty or IDCG@K == 0.
    """
    if not relevant_chunk_ids or k <= 0 or not retrieved_chunk_ids:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    top_k_retrieved = retrieved_chunk_ids[:k]

    dcg = 0.0
    for idx, chunk_id in enumerate(top_k_retrieved, start=1):
        if chunk_id in relevant_set:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0

    return round(dcg / idcg, 4)


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


def answer_token_f1(generated_answer: str, ground_truth_answer: Optional[str]) -> float:
    """
    Calculate token-level F1 score between generated answer and ground truth answer.
    F1 = 2 * (precision * recall) / (precision + recall)
    Returns 0.0 if either string is empty or no tokens match.
    """
    if not generated_answer or not ground_truth_answer:
        return 0.0

    gen_tokens = normalize_text(generated_answer, remove_stopwords=True)
    gt_tokens = normalize_text(ground_truth_answer, remove_stopwords=True)

    if not gen_tokens or not gt_tokens:
        return 0.0

    common = Counter(gen_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(gen_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return round(f1, 4)


def context_support_score(generated_answer: str, retrieved_texts: List[str]) -> float:
    """
    Calculate lexical context support score:
    Fraction of non-stopword tokens in generated_answer that appear in retrieved_texts.
    Returns 0.0 if generated_answer is empty or no tokens overlap.
    """
    if not generated_answer or not retrieved_texts:
        return 0.0

    gen_tokens = normalize_text(generated_answer, remove_stopwords=True)
    if not gen_tokens:
        return 0.0

    context_str = " ".join(retrieved_texts)
    context_tokens = set(normalize_text(context_str, remove_stopwords=False))

    hits = sum(1 for token in gen_tokens if token in context_tokens)
    return round(hits / len(gen_tokens), 4)


def citation_coverage_score(citations: List[Any], retrieved_chunk_ids: List[str]) -> float:
    """
    Calculate citation coverage score:
    Fraction of citations that correspond to valid retrieved chunk IDs.
    Returns 1.0 if citations exist and match retrieved chunks, else 0.0.
    """
    if not citations:
        return 0.0

    retrieved_set = set(retrieved_chunk_ids)
    cited_ids = []
    for c in citations:
        chunk_id = getattr(c, "chunk_id", None) or (c.get("chunk_id") if isinstance(c, dict) else None)
        if chunk_id:
            cited_ids.append(chunk_id)

    if not cited_ids:
        return 0.0

    valid_citations = sum(1 for c_id in cited_ids if c_id in retrieved_set)
    return round(valid_citations / len(cited_ids), 4)


def unanswerable_safe_handling(generated_answer: str, has_relevant_context: bool) -> bool:
    """
    Check whether an out-of-corpus / unanswerable question was handled safely.
    Returns True if the system appropriately indicated no answer/context is available,
    or False if it generated a confident response despite lacking context.
    """
    if not has_relevant_context:
        return True

    if not generated_answer or not generated_answer.strip():
        return True

    safe_phrases = [
        "not available",
        "no relevant",
        "not provided",
        "cannot be answered",
        "insufficient",
        "does not contain",
        "unanswerable",
        "not mentioned",
        "no information",
    ]
    gen_lower = generated_answer.lower()
    return any(phrase in gen_lower for phrase in safe_phrases)
