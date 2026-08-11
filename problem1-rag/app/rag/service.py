import logging
from typing import Any, Dict, List, Optional

from app.generation.base import BaseGenerator
from app.retrieval.base import BaseRetriever
from app.schemas.document import DocumentChunk
from app.schemas.query import RAGQueryResponse

logger = logging.getLogger(__name__)


class RAGService:
    """
    Thin RAG orchestration service.
    Connects abstract retrieval (BaseRetriever) and generation (BaseGenerator) layers
    without directly depending on storage or LLM implementation details.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
    ):
        self.retriever = retriever
        self.generator = generator

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RAGQueryResponse:
        """
        Orchestrate RAG workflow:
        1. Retrieve candidate DocumentChunk objects via injected retriever.
        2. Pass query and retrieved chunks to injected generator.
        3. Return resulting RAGQueryResponse.
        """
        logger.info(f"RAGService processing query: '{query[:50]}...'")

        # 1. Retrieve chunks via injected retriever
        chunks: List[DocumentChunk] = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            metadata_filter=metadata_filter,
        )

        logger.info(f"RAGService retrieved {len(chunks)} chunks.")

        # 2. Generate answer via injected generator
        response: RAGQueryResponse = self.generator.generate(
            query=query,
            context_chunks=chunks,
        )

        return response
