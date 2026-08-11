import hashlib
from typing import List, Optional

from app.config.settings import get_settings
from app.ingestion.base import BaseChunker
from app.ingestion.exceptions import InvalidChunkConfigError
from app.schemas.document import Document, DocumentChunk


def _generate_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    """Generate a deterministic sha256 chunk ID based on parent doc_id, chunk index, and chunk text."""
    hasher = hashlib.sha256()
    hasher.update(f"{doc_id}:chunk_{chunk_index}:{chunk_text}".encode("utf-8"))
    return hasher.hexdigest()[:32]


class TextChunker(BaseChunker):
    """
    Deterministic sliding-window text chunker.
    Splits documents into overlapping text chunks respecting chunk_size and chunk_overlap settings.
    """

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate chunk size and overlap parameters."""
        if self.chunk_size <= 0:
            raise InvalidChunkConfigError(f"chunk_size must be strictly positive (> 0), got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise InvalidChunkConfigError(f"chunk_overlap must be non-negative (>= 0), got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise InvalidChunkConfigError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a Document into deterministic DocumentChunk objects."""
        content = document.content.strip()
        if not content:
            return []

        # Short text case
        if len(content) <= self.chunk_size:
            chunk_id = _generate_chunk_id(document.doc_id, 0, content)
            metadata = dict(document.metadata)
            metadata.update({
                "chunk_index": 0,
                "chunk_size": len(content),
            })
            return [
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    text=content,
                    chunk_index=0,
                    metadata=metadata,
                )
            ]

        chunks: List[DocumentChunk] = []
        start = 0
        chunk_index = 0
        content_length = len(content)
        step = self.chunk_size - self.chunk_overlap

        while start < content_length:
            end = start + self.chunk_size

            # If end is before the end of string, try to snap to natural word/line boundary
            if end < content_length:
                # Look backwards for a newline or space within a search window
                boundary = max(content.rfind("\n", start, end), content.rfind(" ", start, end))
                if boundary > start + (self.chunk_size // 2):
                    end = boundary

            chunk_text = content[start:end].strip()

            if chunk_text:
                chunk_id = _generate_chunk_id(document.doc_id, chunk_index, chunk_text)
                metadata = dict(document.metadata)
                metadata.update({
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_text),
                })

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

            if end >= content_length:
                break

            # Advance start for the next chunk, respecting overlap
            next_start = end - self.chunk_overlap
            # Ensure progress to avoid infinite loops
            if next_start <= start:
                next_start = start + max(1, step)
            start = next_start

        return chunks
