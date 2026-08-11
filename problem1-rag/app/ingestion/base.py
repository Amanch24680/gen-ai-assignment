from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from app.schemas.document import Document, DocumentChunk


class BaseDocumentParser(ABC):
    """Abstract base class for document parsers (PDF, HTML, Markdown)."""

    @abstractmethod
    def parse(self, file_path: Path) -> List[Document]:
        """
        Parse a document file and return a list of Document objects.
        For page-based files (like PDFs), returns one Document per page.
        For single-page files (like HTML/Markdown), returns a list with one Document.
        """
        pass


class BaseChunker(ABC):
    """Abstract base class for document text chunking strategies."""

    @abstractmethod
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a Document into smaller deterministic DocumentChunk segments."""
        pass
