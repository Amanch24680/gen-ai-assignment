from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from app.schemas.document import DocumentChunk


class BaseVectorStore(ABC):
    """Abstract base class for vector store operations (indexing, search)."""

    @abstractmethod
    def upsert_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Upsert embedded document chunks into the vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int,
        relevance_threshold: float,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Search vector store for most similar chunks matching filter and threshold."""
        pass
