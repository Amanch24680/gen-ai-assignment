import pytest
from evaluation.metrics import (
    answer_token_f1,
    citation_coverage_score,
    context_support_score,
    normalize_text,
    unanswerable_safe_handling,
)


def test_normalize_text():
    assert normalize_text("Hello, World!", remove_stopwords=False) == ["hello", "world"]
    assert normalize_text("The vector store is Qdrant.", remove_stopwords=True) == ["vector", "store", "qdrant"]
    assert normalize_text("", remove_stopwords=True) == []


def test_answer_token_f1_perfect_match():
    gen = "The suggested low-cost vector stores are pgvector, Qdrant, ChromaDB, LanceDB, FAISS, and sqlite-vec."
    gt = "The suggested low-cost vector stores are pgvector, Qdrant (self-hosted/embedded), ChromaDB, LanceDB, FAISS, and sqlite-vec."
    score = answer_token_f1(gen, gt)
    assert score > 0.85


def test_answer_token_f1_partial_match():
    gen = "Qdrant and ChromaDB vector stores."
    gt = "pgvector, Qdrant, ChromaDB, LanceDB, FAISS, sqlite-vec."
    score = answer_token_f1(gen, gt)
    assert 0.0 < score < 1.0


def test_answer_token_f1_empty_edge_cases():
    assert answer_token_f1("", "reference answer") == 0.0
    assert answer_token_f1("generated answer", None) == 0.0
    assert answer_token_f1("", None) == 0.0


def test_context_support_score():
    gen = "BAAI/bge-small-en-v1.5 and Gemma 2B model."
    retrieved_texts = [
        "The model is BAAI/bge-small-en-v1.5 embedding and quantized Gemma 2B generator."
    ]
    score = context_support_score(gen, retrieved_texts)
    assert score > 0.8

    assert context_support_score("", retrieved_texts) == 0.0
    assert context_support_score(gen, []) == 0.0


def test_citation_coverage_score():
    class DummyCitation:
        def __init__(self, chunk_id):
            self.chunk_id = chunk_id

    citations = [DummyCitation("c1"), DummyCitation("c2")]
    retrieved_chunk_ids = ["c1", "c2", "c3"]

    assert citation_coverage_score(citations, retrieved_chunk_ids) == 1.0
    assert citation_coverage_score([], retrieved_chunk_ids) == 0.0


def test_unanswerable_safe_handling():
    # Context unavailable -> safe
    assert unanswerable_safe_handling("Some answer", has_relevant_context=False) is True

    # Context available + phrase indicating unavailable -> safe
    assert unanswerable_safe_handling("The information is not available in the corpus.", has_relevant_context=True) is True

    # Context available + confident answer -> unsafe
    assert unanswerable_safe_handling("The GPU VRAM required for 70B LLM is 140GB.", has_relevant_context=True) is False
