from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from app.schemas.document import DocumentChunk


class BaseRetriever(ABC):
    """Abstract base class for high-level document retrieval logic."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Retrieve relevant context chunks for a user query."""
        pass
