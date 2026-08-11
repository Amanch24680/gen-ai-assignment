from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from app.schemas.document import Document, DocumentChunk


class BaseDocumentParser(ABC):
    """Abstract base class for document parsers (PDF, HTML, Markdown)."""

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """Parse a document file into a unified Document schema."""
        pass


class BaseChunker(ABC):
    """Abstract base class for document text chunking strategies."""

    @abstractmethod
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a Document into smaller DocumentChunk segments."""
        pass
