from typing import Optional
from pydantic import BaseModel, Field


class QueryExecutionMetrics(BaseModel):
    """Observability metrics schema for query latency, token usage, and chunk stats."""
    retrieval_latency_ms: float = Field(description="Retrieval step latency in milliseconds")
    generation_latency_ms: float = Field(description="LLM generation step latency in milliseconds")
    total_latency_ms: float = Field(description="Total endpoint latency in milliseconds")
    retrieved_chunk_count: int = Field(description="Count of chunks retrieved above threshold")
    prompt_tokens: Optional[int] = Field(default=None, description="Number of tokens in prompt")
    completion_tokens: Optional[int] = Field(default=None, description="Number of tokens in generated response")
    total_tokens: Optional[int] = Field(default=None, description="Total token count for request")
