from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.metrics import QueryExecutionMetrics


class Citation(BaseModel):
    """Citation metadata for a grounded answer segment."""
    doc_id: str = Field(description="Source document ID")
    chunk_id: str = Field(description="Source chunk ID")
    snippet: str = Field(description="Excerpt of text cited from chunk")
    score: float = Field(description="Relevance similarity score")


class RAGQueryRequest(BaseModel):
    """HTTP API Request model for RAG generation."""
    query: str = Field(description="User input query")
    top_k: Optional[int] = Field(default=None, description="Optional override for top_k retrieval")
    relevance_threshold: Optional[float] = Field(default=None, description="Optional override for relevance threshold")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter dictionary")


class RAGQueryResponse(BaseModel):
    """HTTP API Response model for RAG generation."""
    query: str = Field(description="Original user query")
    answer: str = Field(description="Grounded LLM generated answer")
    citations: List[Citation] = Field(default_factory=list, description="List of document citations")
    retrieved_chunk_count: int = Field(description="Number of chunks retrieved above relevance threshold")
    has_relevant_context: bool = Field(description="Flag indicating if relevant context was found")
    latency_ms: float = Field(description="Total request execution latency in milliseconds")
    prompt_tokens: Optional[int] = Field(default=None, description="Number of tokens in prompt")
    completion_tokens: Optional[int] = Field(default=None, description="Number of tokens in generated response")
    total_tokens: Optional[int] = Field(default=None, description="Total token count for request")
    metrics: Optional[QueryExecutionMetrics] = Field(default=None, description="Detailed execution & observability metrics")
