from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Raw parsed document container."""
    doc_id: str = Field(description="Unique identifier for the document")
    content: str = Field(description="Full extracted text content")
    file_type: str = Field(description="Format of document (pdf, html, md)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated document metadata")


class DocumentChunk(BaseModel):
    """Chunked document segment with optional embedding vector."""
    chunk_id: str = Field(description="Unique identifier for the chunk")
    doc_id: str = Field(description="Parent document identifier")
    text: str = Field(description="Chunk text content")
    chunk_index: int = Field(description="Sequential position of chunk within document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk level metadata")
    embedding: Optional[list[float]] = Field(default=None, description="Vector embedding representation")
