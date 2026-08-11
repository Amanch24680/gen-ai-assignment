from pathlib import Path
import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.parsers import DocumentLoader


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("fixture_filename", ["sample.pdf", "sample.html", "sample.md"])
def test_ingestion_and_chunking_idempotency(fixtures_dir, fixture_filename):
    """
    Process the same document twice and verify:
    - Same document_id
    - Same chunk IDs
    - Same chunk contents
    - Same metadata
    """
    file_path = fixtures_dir / fixture_filename
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)

    # First pass
    docs_pass1 = loader.load_document(file_path)
    chunks_pass1 = []
    for doc in docs_pass1:
        chunks_pass1.extend(chunker.chunk_document(doc))

    # Second pass
    docs_pass2 = loader.load_document(file_path)
    chunks_pass2 = []
    for doc in docs_pass2:
        chunks_pass2.extend(chunker.chunk_document(doc))

    # Verify Document-level idempotency
    assert len(docs_pass1) == len(docs_pass2)
    for d1, d2 in zip(docs_pass1, docs_pass2):
        assert d1.doc_id == d2.doc_id
        assert d1.content == d2.content
        assert d1.file_type == d2.file_type

    # Verify Chunk-level idempotency
    assert len(chunks_pass1) == len(chunks_pass2)
    for c1, c2 in zip(chunks_pass1, chunks_pass2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.doc_id == c2.doc_id
        assert c1.text == c2.text
        assert c1.chunk_index == c2.chunk_index
        assert c1.metadata == c2.metadata
