from pathlib import Path
import pytest

from app.ingestion.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.ingestion.parsers import (
    DocumentLoader,
    HTMLParser,
    MarkdownParser,
    PDFParser,
)


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def test_pdf_parser_success(fixtures_dir):
    """Test loading and parsing a valid multi-page PDF document."""
    pdf_path = fixtures_dir / "sample.pdf"
    parser = PDFParser()
    docs = parser.parse(pdf_path)

    assert len(docs) == 2
    assert docs[0].file_type == "pdf"
    assert docs[0].metadata["page_number"] == 1
    assert "page 1" in docs[0].content
    assert docs[1].metadata["page_number"] == 2
    assert "page 2" in docs[1].content
    assert docs[0].doc_id != docs[1].doc_id


def test_pdf_parser_corrupt_file(fixtures_dir):
    """Test handling of corrupt PDF files."""
    corrupt_path = fixtures_dir / "corrupt.pdf"
    parser = PDFParser()
    with pytest.raises(DocumentParsingError) as exc_info:
        parser.parse(corrupt_path)
    assert "Failed to read PDF file" in str(exc_info.value) or "PdfStreamError" in str(exc_info.value)


def test_html_parser_success(fixtures_dir):
    """Test loading and parsing an HTML document, verifying markup stripping."""
    html_path = fixtures_dir / "sample.html"
    parser = HTMLParser()
    docs = parser.parse(html_path)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.file_type == "html"
    assert "Cost-Efficient RAG Architecture" in doc.content
    assert "ignore script" not in doc.content  # Script content removed
    assert doc.metadata["page_number"] is None
    assert doc.metadata["filename"] == "sample.html"


def test_markdown_parser_success(fixtures_dir):
    """Test loading and parsing a Markdown document."""
    md_path = fixtures_dir / "sample.md"
    parser = MarkdownParser()
    docs = parser.parse(md_path)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.file_type == "md"
    assert "Cost-Efficient RAG System Specification" in doc.content
    assert "Qdrant running in local mode" in doc.content
    assert doc.metadata["page_number"] is None


def test_document_loader_dispatch(fixtures_dir):
    """Test DocumentLoader dispatches correctly based on file extensions."""
    loader = DocumentLoader()

    pdf_docs = loader.load_document(fixtures_dir / "sample.pdf")
    assert len(pdf_docs) == 2

    html_docs = loader.load_document(fixtures_dir / "sample.html")
    assert len(html_docs) == 1

    md_docs = loader.load_document(fixtures_dir / "sample.md")
    assert len(md_docs) == 1


def test_unsupported_file_type(tmp_path):
    """Test loading an unsupported file extension raises UnsupportedFileTypeError."""
    unsupported_file = tmp_path / "test.docx"
    unsupported_file.write_text("Test content", encoding="utf-8")

    loader = DocumentLoader()
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        loader.load_document(unsupported_file)
    assert "Unsupported file type '.docx'" in str(exc_info.value)


def test_empty_document_error(tmp_path):
    """Test loading an empty document raises EmptyDocumentError."""
    empty_md = tmp_path / "empty.md"
    empty_md.write_text("", encoding="utf-8")

    loader = DocumentLoader()
    with pytest.raises(EmptyDocumentError):
        loader.load_document(empty_md)
