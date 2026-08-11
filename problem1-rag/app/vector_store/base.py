from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from app.schemas.document import DocumentChunk


class BaseVectorStore(ABC):
    """Abstract base class for vector store operations (indexing, search, administration)."""

    @abstractmethod
    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Upsert embedded document chunks into the vector store and return the number of points processed."""
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

    @abstractmethod
    def count_chunks(self) -> int:
        """Return total count of chunks in the vector store collection."""
        pass

    @abstractmethod
    def reset_collection(self) -> None:
        """Reset/clear the collection for testing or re-indexing."""
        pass
