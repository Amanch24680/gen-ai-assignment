import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.ingestion.base import BaseDocumentParser
from app.ingestion.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.schemas.document import Document


def _normalize_text(text: str) -> str:
    """Clean and normalize whitespace in extracted text."""
    if not text:
        return ""
    # Replace carriage returns and weird space characters
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple inline spaces to a single space
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _generate_doc_id(stable_identifier: str, page_or_section: str, content: str) -> str:
    """Generate a portable deterministic sha256 document ID based on filename/identifier, page/section, and content."""
    hasher = hashlib.sha256()
    hasher.update(f"{stable_identifier}:{page_or_section}:{content}".encode("utf-8"))
    return hasher.hexdigest()[:32]


class PDFParser(BaseDocumentParser):
    """Parser for PDF documents using pypdf."""

    def parse(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise DocumentParsingError(f"PDF file does not exist: {file_path}")

        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            raise DocumentParsingError(f"Failed to read PDF file '{file_path.name}': {e}") from e

        if not reader.pages:
            raise EmptyDocumentError(f"PDF file contains no pages: {file_path.name}")

        documents: List[Document] = []
        total_pages = len(reader.pages)
        source_str = str(file_path)

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                raise DocumentParsingError(f"Failed to extract text from page {page_idx} in '{file_path.name}': {e}") from e

            normalized_content = _normalize_text(raw_text)
            if not normalized_content:
                continue

            doc_id = _generate_doc_id(file_path.name, f"page_{page_idx}", normalized_content)
            metadata = {
                "filename": file_path.name,
                "source": source_str,
                "file_type": "pdf",
                "page_number": page_idx,
                "total_pages": total_pages,
            }

            documents.append(
                Document(
                    doc_id=doc_id,
                    content=normalized_content,
                    file_type="pdf",
                    metadata=metadata,
                )
            )

        if not documents:
            raise EmptyDocumentError(f"PDF file contains no extractable text: {file_path.name}")

        return documents


class HTMLParser(BaseDocumentParser):
    """Parser for HTML documents using BeautifulSoup."""

    def parse(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise DocumentParsingError(f"HTML file does not exist: {file_path}")

        try:
            raw_html = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise DocumentParsingError(f"Failed to read HTML file '{file_path.name}': {e}") from e

        try:
            soup = BeautifulSoup(raw_html, "html.parser")
            # Remove script, style, and navigation tags
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            raw_text = soup.get_text(separator="\n")
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse HTML in '{file_path.name}': {e}") from e

        normalized_content = _normalize_text(raw_text)
        if not normalized_content:
            raise EmptyDocumentError(f"HTML file contains no extractable text: {file_path.name}")

        source_str = str(file_path)
        doc_id = _generate_doc_id(file_path.name, "main", normalized_content)
        metadata = {
            "filename": file_path.name,
            "source": source_str,
            "file_type": "html",
            "page_number": None,
        }

        return [
            Document(
                doc_id=doc_id,
                content=normalized_content,
                file_type="html",
                metadata=metadata,
            )
        ]


class MarkdownParser(BaseDocumentParser):
    """Parser for Markdown documents."""

    def parse(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise DocumentParsingError(f"Markdown file does not exist: {file_path}")

        try:
            raw_md = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise DocumentParsingError(f"Failed to read Markdown file '{file_path.name}': {e}") from e

        # Convert markdown headers/formatting to plain text clean lines
        # Remove code blocks ticks, heading # characters, etc for clean indexing
        cleaned_lines = []
        for line in raw_md.splitlines():
            line_str = re.sub(r"^#+\s*", "", line)  # Strip headings
            line_str = re.sub(r"[`*_~]", "", line_str)  # Strip inline formatting
            cleaned_lines.append(line_str)

        raw_text = "\n".join(cleaned_lines)
        normalized_content = _normalize_text(raw_text)
        if not normalized_content:
            raise EmptyDocumentError(f"Markdown file contains no extractable text: {file_path.name}")

        source_str = str(file_path)
        doc_id = _generate_doc_id(file_path.name, "main", normalized_content)
        metadata = {
            "filename": file_path.name,
            "source": source_str,
            "file_type": "md",
            "page_number": None,
        }

        return [
            Document(
                doc_id=doc_id,
                content=normalized_content,
                file_type="md",
                metadata=metadata,
            )
        ]


class DocumentLoader:
    """Unified document loader that dispatches files to appropriate parsers based on file extension."""

    def __init__(self):
        self._parsers = {
            ".pdf": PDFParser(),
            ".html": HTMLParser(),
            ".htm": HTMLParser(),
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
        }

    def load_document(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load and parse a document file, returning a list of Document objects.
        Raises UnsupportedFileTypeError if file extension is not supported.
        Raises DocumentParsingError / EmptyDocumentError for unreadable/empty files.
        """
        path = Path(file_path).resolve()
        ext = path.suffix.lower()
        if ext not in self._parsers:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}' for file '{path.name}'. "
                f"Supported types are: {list(self._parsers.keys())}"
            )

        parser = self._parsers[ext]
        return parser.parse(path)
