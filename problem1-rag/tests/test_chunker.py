import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.exceptions import InvalidChunkConfigError
from app.schemas.document import Document


def test_short_text_single_chunk():
    """Verify short text under chunk_size creates exactly one chunk."""
    doc = Document(
        doc_id="doc_short",
        content="Short text content for testing single chunk creation.",
        file_type="md",
        metadata={"filename": "test.md"},
    )
    chunker = TextChunker(chunk_size=800, chunk_overlap=120)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].doc_id == "doc_short"
    assert chunks[0].text == doc.content
    assert chunks[0].metadata["filename"] == "test.md"


def test_long_text_multiple_chunks_with_overlap():
    """Verify long text creates multiple chunks with overlap present between adjacent chunks."""
    long_content = (
        "Retrieval-Augmented Generation (RAG) is an AI framework for retrieving facts from an external knowledge base "
        "to ground LLMs on the most accurate, up-to-date information. " * 15
    )
    doc = Document(
        doc_id="doc_long",
        content=long_content,
        file_type="pdf",
        metadata={"filename": "long.pdf"},
    )

    chunk_size = 200
    chunk_overlap = 50
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1

    for i in range(len(chunks)):
        assert chunks[i].chunk_index == i
        assert chunks[i].doc_id == "doc_long"
        assert len(chunks[i].text) > 0  # No empty chunks

    # Check overlap between adjacent chunks: tail of chunk i should overlap with head of chunk i+1
    for i in range(len(chunks) - 1):
        c1_text = chunks[i].text
        c2_text = chunks[i + 1].text
        c1_tail = c1_text[-chunk_overlap:]
        # At least a portion of the tail of c1 must appear near the start of c2
        shared_words = [w for w in c1_tail.split() if len(w) > 3]
        if shared_words:
            assert any(w in c2_text[:chunk_overlap + 30] for w in shared_words)


def test_no_empty_chunks():
    """Verify chunker never produces empty chunks."""
    doc = Document(
        doc_id="doc_spaces",
        content="   \n\n   Word1 Word2 Word3   \n\n   ",
        file_type="md",
        metadata={},
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "Word1 Word2 Word3"


def test_deterministic_chunk_ids():
    """Verify chunk IDs are deterministic for identical input."""
    doc = Document(
        doc_id="doc_fixed",
        content="Deterministic chunking text validation content.",
        file_type="md",
        metadata={},
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    chunks1 = chunker.chunk_document(doc)
    chunks2 = chunker.chunk_document(doc)

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text


@pytest.mark.parametrize("invalid_size", [0, -10])
def test_invalid_chunk_size_rejected(invalid_size):
    """Verify invalid non-positive chunk_size raises InvalidChunkConfigError."""
    with pytest.raises(InvalidChunkConfigError):
        TextChunker(chunk_size=invalid_size, chunk_overlap=10)


@pytest.mark.parametrize("invalid_overlap", [-1, -50])
def test_invalid_negative_overlap_rejected(invalid_overlap):
    """Verify negative chunk_overlap raises InvalidChunkConfigError."""
    with pytest.raises(InvalidChunkConfigError):
        TextChunker(chunk_size=500, chunk_overlap=invalid_overlap)


@pytest.mark.parametrize("size,overlap", [(200, 200), (200, 250)])
def test_overlap_greater_or_equal_to_size_rejected(size, overlap):
    """Verify overlap >= chunk_size raises InvalidChunkConfigError."""
    with pytest.raises(InvalidChunkConfigError):
        TextChunker(chunk_size=size, chunk_overlap=overlap)
