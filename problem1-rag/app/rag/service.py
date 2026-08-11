import logging
import time
from typing import Any, Dict, List, Optional

from app.generation.base import BaseGenerator
from app.logging.logger import BaseQueryLogger, QueryObservabilityLogger
from app.retrieval.base import BaseRetriever
from app.schemas.document import DocumentChunk
from app.schemas.metrics import QueryExecutionMetrics
from app.schemas.query import RAGQueryResponse

logger = logging.getLogger(__name__)


class RAGService:
    """
    Thin RAG orchestration service with observability.
    Connects abstract retrieval (BaseRetriever) and generation (BaseGenerator) layers
    and records execution metrics via QueryObservabilityLogger.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        query_logger: Optional[BaseQueryLogger] = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.query_logger = query_logger or QueryObservabilityLogger()

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RAGQueryResponse:
        """
        Orchestrate RAG workflow with metrics tracking:
        1. Measure and perform document chunk retrieval.
        2. Measure and perform answer generation.
        3. Construct QueryExecutionMetrics and log via query_logger.
        4. Return populated RAGQueryResponse.
        """
        logger.info(f"RAGService processing query: '{query[:50]}...'")
        start_total = time.perf_counter()

        # 1. Retrieve chunks via injected retriever
        start_retrieval = time.perf_counter()
        chunks: List[DocumentChunk] = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            metadata_filter=metadata_filter,
        )
        retrieval_latency_ms = (time.perf_counter() - start_retrieval) * 1000.0
        logger.info(f"RAGService retrieved {len(chunks)} chunks in {retrieval_latency_ms:.2f}ms.")

        # 2. Generate answer via injected generator
        start_gen = time.perf_counter()
        response: RAGQueryResponse = self.generator.generate(
            query=query,
            context_chunks=chunks,
        )
        generation_latency_ms = (time.perf_counter() - start_gen) * 1000.0
        total_latency_ms = (time.perf_counter() - start_total) * 1000.0

        # 3. Construct QueryExecutionMetrics
        metrics = QueryExecutionMetrics(
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            retrieved_chunk_count=len(chunks),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )

        # 4. Attach metrics to response & update total latency
        response.metrics = metrics
        response.latency_ms = round(total_latency_ms, 2)

        # 5. Log metrics via query_logger
        if self.query_logger:
            try:
                self.query_logger.log_query_metrics(query, metrics)
            except Exception as exc:
                logger.warning(f"Failed to log query metrics: {exc}")

        return response
