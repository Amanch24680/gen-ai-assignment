import logging
from abc import ABC, abstractmethod
from app.schemas.metrics import QueryExecutionMetrics

logger = logging.getLogger("problem1_rag.observability")


class BaseQueryLogger(ABC):
    """Abstract base class for query execution logging and observability."""

    @abstractmethod
    def log_query_metrics(self, query: str, metrics: QueryExecutionMetrics) -> None:
        """Log execution metrics including latency, chunk count, and token usage."""
        pass


class QueryObservabilityLogger(BaseQueryLogger):
    """Standard logger implementation for query metrics."""

    def log_query_metrics(self, query: str, metrics: QueryExecutionMetrics) -> None:
        logger.info(
            f"Query Metrics | Query: '{query}' | "
            f"Total Latency: {metrics.total_latency_ms:.2f}ms | "
            f"Retrieval Latency: {metrics.retrieval_latency_ms:.2f}ms | "
            f"Generation Latency: {metrics.generation_latency_ms:.2f}ms | "
            f"Retrieved Chunks: {metrics.retrieved_chunk_count} | "
            f"Total Tokens: {metrics.total_tokens}"
        )
