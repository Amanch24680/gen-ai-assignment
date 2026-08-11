from typing import Any, Dict, List, Optional
import pytest

from app.embeddings.base import BaseEmbeddingService
from app.embeddings.service import SentenceTransformerEmbeddingService
from app.retrieval.exceptions import EmptyQueryError, InvalidRetrievalConfigError
from app.retrieval.service import VectorRetriever
from app.schemas.document import DocumentChunk
from app.vector_store.base import BaseVectorStore
from app.vector_store.qdrant_store import QdrantVectorStore


class DummyMockEmbeddingService(BaseEmbeddingService):
    """Mock embedding service for fast, deterministic unit testing without model inference."""

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension
        # Generate deterministic vector based on text hash
        val = (hash(text) % 100) / 100.0
        return [val] * self.dimension

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class DummyMockVectorStore(BaseVectorStore):
    """Mock vector store for unit testing retrieval filtering, top_k, and tie-breaking."""

    def __init__(self, stored_chunks: Optional[List[DocumentChunk]] = None):
        self.stored_chunks = stored_chunks if stored_chunks is not None else []

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        self.stored_chunks.extend(chunks)
        return len(chunks)

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        relevance_threshold: float,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        results = []
        for chunk in self.stored_chunks:
            # Score attached or simulated
            score = chunk.score if chunk.score is not None else chunk.metadata.get("score", 0.0)
            if score >= relevance_threshold:
                # Optionally filter metadata
                if metadata_filter:
                    match = all(chunk.metadata.get(k) == v for k, v in metadata_filter.items())
                    if not match:
                        continue
                results.append(chunk)
        return results[:top_k]

    def count_chunks(self) -> int:
        return len(self.stored_chunks)

    def reset_collection(self) -> None:
        self.stored_chunks = []


# --- Unit Tests ---

def test_retrieval_empty_query_rejection():
    """Unit test: Empty query string must raise EmptyQueryError."""
    retriever = VectorRetriever(
        embedding_service=DummyMockEmbeddingService(),
        vector_store=DummyMockVectorStore(),
    )

    with pytest.raises(EmptyQueryError):
        retriever.retrieve("")


def test_retrieval_whitespace_query_rejection():
    """Unit test: Whitespace-only query string must raise EmptyQueryError."""
    retriever = VectorRetriever(
        embedding_service=DummyMockEmbeddingService(),
        vector_store=DummyMockVectorStore(),
    )

    with pytest.raises(EmptyQueryError):
        retriever.retrieve("     \t\n  ")


def test_retrieval_invalid_config_parameters():
    """Unit test: Invalid top_k or relevance_threshold parameters raise InvalidRetrievalConfigError."""
    retriever = VectorRetriever(
        embedding_service=DummyMockEmbeddingService(),
        vector_store=DummyMockVectorStore(),
    )

    with pytest.raises(InvalidRetrievalConfigError):
        retriever.retrieve("valid query", top_k=0)

    with pytest.raises(InvalidRetrievalConfigError):
        retriever.retrieve("valid query", relevance_threshold=1.5)


def test_retrieval_top_k_limiting_and_relevance_filtering():
    """Unit test: Verify relevance threshold excludes low scores and top_k caps result count."""
    chunks = [
        DocumentChunk(
            chunk_id="chunk_high",
            doc_id="doc_1",
            text="High relevance content",
            chunk_index=0,
            metadata={"filename": "doc1.pdf", "score": 0.85},
            score=0.85,
        ),
        DocumentChunk(
            chunk_id="chunk_mid",
            doc_id="doc_1",
            text="Medium relevance content",
            chunk_index=1,
            metadata={"filename": "doc1.pdf", "score": 0.50},
            score=0.50,
        ),
        DocumentChunk(
            chunk_id="chunk_low",
            doc_id="doc_2",
            text="Low relevance content",
            chunk_index=0,
            metadata={"filename": "doc2.pdf", "score": 0.20},
            score=0.20,
        ),
    ]

    vector_store = DummyMockVectorStore(stored_chunks=chunks)
    retriever = VectorRetriever(
        embedding_service=DummyMockEmbeddingService(),
        vector_store=vector_store,
    )

    # Threshold 0.35 -> should filter out chunk_low (0.20)
    results = retriever.retrieve("test query", top_k=5, relevance_threshold=0.35)
    assert len(results) == 2
    assert [c.chunk_id for c in results] == ["chunk_high", "chunk_mid"]

    # Cap top_k = 1
    results_top1 = retriever.retrieve("test query", top_k=1, relevance_threshold=0.35)
    assert len(results_top1) == 1
    assert results_top1[0].chunk_id == "chunk_high"


def test_retrieval_no_result_behavior():
    """Unit test: If no chunks meet threshold or store is empty, return []."""
    chunks = [
        DocumentChunk(
            chunk_id="chunk_low",
            doc_id="doc_1",
            text="Below threshold text",
            chunk_index=0,
            metadata={"filename": "doc1.pdf", "score": 0.15},
            score=0.15,
        )
    ]
    retriever = VectorRetriever(
        embedding_service=DummyMockEmbeddingService(),
        vector_store=DummyMockVectorStore(stored_chunks=chunks),
    )

    results = retriever.retrieve("test query", relevance_threshold=0.50)
    assert results == []


def test_retrieval_deterministic_ordering_and_tie_breaking():
    """Unit test: Verify score descending order and chunk_id ascending tie-breaker."""
    chunks = [
        DocumentChunk(
            chunk_id="chunk_b",
            doc_id="doc_1",
            text="Tie score chunk B",
            chunk_index=1,
            metadata={"score": 0.80},
            score=0.80,
        ),
        DocumentChunk(
            chunk_id="chunk_a",
            doc_id="doc_1",
            text="Tie score chunk A",
            chunk_index=0,
            metadata={"score": 0.80},
            score=0.80,
        ),
        DocumentChunk(
            chunk_id="chunk_top",
            doc_id="doc_2",
            text="Top score chunk",
            chunk_index=0,
            metadata={"score": 0.95},
            score=0.95,
        ),
    ]

    retriever = VectorRetriever(
        embedding_service=DummyMockEmbeddingService(),
        vector_store=DummyMockVectorStore(stored_chunks=chunks),
    )

    results = retriever.retrieve("test query", relevance_threshold=0.0)
    assert len(results) == 3
    # Top score 0.95 first, then chunk_a (0.80) before chunk_b (0.80) alphabetically
    assert [c.chunk_id for c in results] == ["chunk_top", "chunk_a", "chunk_b"]


# --- Integration Tests ---

@pytest.mark.integration
def test_retrieval_real_embedding_and_qdrant_integration(tmp_path):
    """
    Integration test: Real BAAI/bge-small-en-v1.5 embedding + local Qdrant + VectorRetriever.
    - Embeds real chunks
    - Stores in local Qdrant DB
    - Queries semantically
    - Verifies top retrieved result and score preservation
    """
    embed_service = SentenceTransformerEmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    db_path = str(tmp_path / "retrieval_qdrant_db")

    qdrant_store = QdrantVectorStore(
        collection_name="retrieval_test_collection",
        vector_size=embed_service.dimension,
        path=db_path,
    )

    text_rag = "Retrieval-Augmented Generation enhances LLM accuracy using external document retrieval."
    text_cooking = "Baking sourdough bread requires flour, water, salt, and wild yeast starter fermentation."

    vec_rag = embed_service.embed_text(text_rag)
    vec_cooking = embed_service.embed_text(text_cooking)

    chunk_rag = DocumentChunk(
        chunk_id="chunk_rag_001",
        doc_id="doc_rag",
        text=text_rag,
        chunk_index=0,
        metadata={"filename": "rag_overview.md", "source": "/docs/rag_overview.md", "file_type": "md"},
        embedding=vec_rag,
    )

    chunk_cooking = DocumentChunk(
        chunk_id="chunk_cook_001",
        doc_id="doc_cooking",
        text=text_cooking,
        chunk_index=0,
        metadata={"filename": "cooking.md", "source": "/docs/cooking.md", "file_type": "md"},
        embedding=vec_cooking,
    )

    qdrant_store.upsert_chunks([chunk_rag, chunk_cooking])
    assert qdrant_store.count_chunks() == 2

    retriever = VectorRetriever(
        embedding_service=embed_service,
        vector_store=qdrant_store,
    )

    # Execute semantic query
    query = "How does RAG improve language model accuracy using documents?"
    results = retriever.retrieve(
        query=query,
        top_k=2,
        relevance_threshold=0.30,
    )

    assert len(results) >= 1
    top_result = results[0]
    assert top_result.chunk_id == "chunk_rag_001"
    assert top_result.doc_id == "doc_rag"
    assert top_result.metadata["filename"] == "rag_overview.md"
    assert top_result.score is not None
    assert top_result.score > 0.45  # Strong semantic similarity score

    qdrant_store.close()
