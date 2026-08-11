from abc import ABC, abstractmethod
from typing import List
from app.schemas.document import DocumentChunk
from app.schemas.query import RAGQueryResponse


class BaseGenerator(ABC):
    """Abstract base class for grounded RAG generation using local LLM."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context_chunks: List[DocumentChunk]
    ) -> RAGQueryResponse:
        """Generate a grounded answer with citations or respond when no context is relevant."""
        pass
