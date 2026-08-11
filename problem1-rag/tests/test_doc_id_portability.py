from pathlib import Path
import pytest

from app.ingestion.parsers import DocumentLoader, MarkdownParser, PDFParser


def test_doc_id_portable_across_directories(tmp_path):
    """
    Requirement 1: Verify that the same file content and filename in directory A and directory B
    produce the exact same document_id (cross-directory portability).
    """
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()

    file_a = dir_a / "document.md"
    file_b = dir_b / "document.md"

    content = "# Portable Document Content\nThis is identical content in two different folders."
    file_a.write_text(content, encoding="utf-8")
    file_b.write_text(content, encoding="utf-8")

    parser = MarkdownParser()
    doc_a = parser.parse(file_a)[0]
    doc_b = parser.parse(file_b)[0]

    assert doc_a.doc_id == doc_b.doc_id


def test_same_content_different_filename_different_doc_id(tmp_path):
    """
    Requirement 2: Verify that the same file content with different filenames
    produces different document_ids.
    """
    file_1 = tmp_path / "file_one.md"
    file_2 = tmp_path / "file_two.md"

    content = "# Shared Content\nSame text in differently named files."
    file_1.write_text(content, encoding="utf-8")
    file_2.write_text(content, encoding="utf-8")

    parser = MarkdownParser()
    doc_1 = parser.parse(file_1)[0]
    doc_2 = parser.parse(file_2)[0]

    assert doc_1.doc_id != doc_2.doc_id


def test_same_filename_different_content_different_doc_id(tmp_path):
    """
    Requirement 3: Verify that the same filename with different content
    produces different document_ids.
    """
    dir_1 = tmp_path / "dir_1"
    dir_2 = tmp_path / "dir_2"
    dir_1.mkdir()
    dir_2.mkdir()

    file_1 = dir_1 / "doc.md"
    file_2 = dir_2 / "doc.md"

    file_1.write_text("# Initial Version Content", encoding="utf-8")
    file_2.write_text("# Updated Version Content", encoding="utf-8")

    parser = MarkdownParser()
    doc_1 = parser.parse(file_1)[0]
    doc_2 = parser.parse(file_2)[0]

    assert doc_1.doc_id != doc_2.doc_id


def test_multipage_pdf_stable_and_distinct_page_ids():
    """
    Requirement 4: Verify that a multi-page PDF produces stable IDs across repeated runs,
    and different IDs for different pages of the same PDF.
    """
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    pdf_path = fixtures_dir / "sample.pdf"

    parser = PDFParser()
    docs_pass_1 = parser.parse(pdf_path)
    docs_pass_2 = parser.parse(pdf_path)

    assert len(docs_pass_1) == 2
    assert len(docs_pass_2) == 2

    # Repeated processing produces identical IDs
    assert docs_pass_1[0].doc_id == docs_pass_2[0].doc_id
    assert docs_pass_1[1].doc_id == docs_pass_2[1].doc_id

    # Page 1 and Page 2 have distinct document IDs
    assert docs_pass_1[0].doc_id != docs_pass_1[1].doc_id
