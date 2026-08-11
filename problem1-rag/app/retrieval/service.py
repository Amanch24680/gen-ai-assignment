import logging
from typing import Any, Dict, List, Optional

from app.config.settings import get_settings
from app.embeddings.base import BaseEmbeddingService
from app.embeddings.service import SentenceTransformerEmbeddingService
from app.retrieval.base import BaseRetriever
from app.retrieval.exceptions import EmptyQueryError, InvalidRetrievalConfigError
from app.schemas.document import DocumentChunk
from app.vector_store.base import BaseVectorStore
from app.vector_store.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """
    Concrete document retrieval service.
    Orchestrates embedding of user query string and querying Qdrant vector store
    with relevance threshold filtering and deterministic ranking.
    """

    def __init__(
        self,
        embedding_service: Optional[BaseEmbeddingService] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        settings = get_settings()
        self.embedding_service = (
            embedding_service if embedding_service is not None else SentenceTransformerEmbeddingService()
        )
        self.vector_store = (
            vector_store if vector_store is not None else QdrantVectorStore()
        )
        self.default_top_k = settings.top_k
        self.default_relevance_threshold = settings.relevance_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Retrieve relevant document chunks for a natural language query.
        
        Steps:
        1. Validate query string (reject empty/whitespace).
        2. Resolve effective top_k and relevance_threshold (falling back to settings defaults).
        3. Embed query using embedding_service.
        4. Query vector_store for top matches matching threshold and optional metadata_filter.
        5. Apply deterministic sorting (score descending, chunk_id ascending tie-breaker).
        6. Return top_k DocumentChunk objects.
        """
        if not query or not query.strip():
            raise EmptyQueryError("Query string cannot be empty or whitespace-only.")

        effective_top_k = top_k if top_k is not None else self.default_top_k
        effective_threshold = (
            relevance_threshold if relevance_threshold is not None else self.default_relevance_threshold
        )

        if effective_top_k <= 0:
            raise InvalidRetrievalConfigError(f"top_k must be greater than 0, got {effective_top_k}.")
        if not (0.0 <= effective_threshold <= 1.0):
            raise InvalidRetrievalConfigError(
                f"relevance_threshold must be between 0.0 and 1.0, got {effective_threshold}."
            )

        logger.info(
            f"Executing retrieval for query='{query[:50]}...' with top_k={effective_top_k}, "
            f"threshold={effective_threshold}, filter={metadata_filter}"
        )

        # 1. Embed query
        query_vector = self.embedding_service.embed_text(query)

        # 2. Search Qdrant vector store
        chunks = self.vector_store.search(
            query_vector=query_vector,
            top_k=effective_top_k,
            relevance_threshold=effective_threshold,
            metadata_filter=metadata_filter,
        )

        # 3. Deterministic ranking: score descending, then chunk_id ascending as tie-breaker
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                -(c.score if c.score is not None else c.metadata.get("score", 0.0)),
                c.chunk_id,
            ),
        )

        # 4. Cap at effective_top_k
        result_chunks = sorted_chunks[:effective_top_k]
        logger.info(f"Retrieved {len(result_chunks)} document chunks above threshold {effective_threshold}.")
        return result_chunks
