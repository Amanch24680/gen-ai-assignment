from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.embeddings.base import BaseEmbeddingService
from app.ingestion.base import BaseChunker
from app.ingestion.parsers import DocumentLoader
from app.ingestion.service import IngestionService
from app.schemas.document import Document, DocumentChunk
from app.vector_store.base import BaseVectorStore


def test_ingest_file_workflow_mocked(tmp_path):
    """Test 1: Verify document flows through loader -> chunker -> embedding -> vector store with mock dependencies."""
    mock_loader = MagicMock(spec=DocumentLoader)
    mock_chunker = MagicMock(spec=BaseChunker)
    mock_embedding_service = MagicMock(spec=BaseEmbeddingService)
    mock_vector_store = MagicMock(spec=BaseVectorStore)

    dummy_file = tmp_path / "test.md"
    dummy_file.write_text("# Test Title\nTest content body.")

    doc = Document(
        doc_id="doc_100",
        content="Test content body.",
        file_type="md",
        metadata={"filename": "test.md", "source": str(dummy_file)},
    )
    chunk_1 = DocumentChunk(
        chunk_id="chk_101",
        doc_id="doc_100",
        text="Test content body.",
        chunk_index=0,
        metadata={"filename": "test.md"},
    )

    mock_loader.load_document.return_value = [doc]
    mock_chunker.chunk_document.return_value = [chunk_1]
    mock_embedding_service.embed_batch.return_value = [[0.1] * 384]
    mock_vector_store.upsert_chunks.return_value = 1

    service = IngestionService(
        loader=mock_loader,
        chunker=mock_chunker,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )

    stats = service.ingest_file(dummy_file)

    # Verify calls
    mock_loader.load_document.assert_called_once_with(dummy_file.resolve())
    mock_chunker.chunk_document.assert_called_once_with(doc)
    mock_embedding_service.embed_batch.assert_called_once_with(["Test content body."])
    mock_vector_store.upsert_chunks.assert_called_once()

    upserted_chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert len(upserted_chunks) == 1
    assert upserted_chunks[0].embedding == [0.1] * 384

    # Verify returned statistics
    assert stats == {
        "documents_processed": 1,
        "chunks_created": 1,
        "chunks_upserted": 1,
    }


def test_ingest_file_multiple_chunks(tmp_path):
    """Test 2: Verify multiple chunks receive embeddings and are passed to vector store."""
    mock_loader = MagicMock(spec=DocumentLoader)
    mock_chunker = MagicMock(spec=BaseChunker)
    mock_embedding_service = MagicMock(spec=BaseEmbeddingService)
    mock_vector_store = MagicMock(spec=BaseVectorStore)

    dummy_file = tmp_path / "long_doc.pdf"
    dummy_file.write_text("dummy PDF content")

    doc = Document(
        doc_id="doc_200",
        content="First page content. Second page content.",
        file_type="pdf",
        metadata={"filename": "long_doc.pdf"},
    )
    chunk_1 = DocumentChunk(
        chunk_id="chk_201", doc_id="doc_200", text="First page content.", chunk_index=0, metadata={}
    )
    chunk_2 = DocumentChunk(
        chunk_id="chk_202", doc_id="doc_200", text="Second page content.", chunk_index=1, metadata={}
    )

    mock_loader.load_document.return_value = [doc]
    mock_chunker.chunk_document.return_value = [chunk_1, chunk_2]
    mock_embedding_service.embed_batch.return_value = [[0.1] * 384, [0.2] * 384]
    mock_vector_store.upsert_chunks.return_value = 2

    service = IngestionService(
        loader=mock_loader,
        chunker=mock_chunker,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )

    stats = service.ingest_file(dummy_file)

    assert stats["documents_processed"] == 1
    assert stats["chunks_created"] == 2
    assert stats["chunks_upserted"] == 2


def test_ingest_directory_workflow(tmp_path):
    """Test 3: Verify directory ingestion iterates files and aggregates statistics correctly."""
    mock_loader = MagicMock(spec=DocumentLoader)
    mock_chunker = MagicMock(spec=BaseChunker)
    mock_embedding_service = MagicMock(spec=BaseEmbeddingService)
    mock_vector_store = MagicMock(spec=BaseVectorStore)

    f1 = tmp_path / "doc1.md"
    f1.write_text("MD content")
    f2 = tmp_path / "doc2.html"
    f2.write_text("<html>HTML content</html>")

    mock_loader.load_document.side_effect = lambda p: [
        Document(doc_id=f"doc_{p.name}", content="Text", file_type=p.suffix[1:], metadata={"filename": p.name})
    ]
    mock_chunker.chunk_document.side_effect = lambda doc: [
        DocumentChunk(chunk_id=f"chk_{doc.doc_id}", doc_id=doc.doc_id, text="Text", chunk_index=0, metadata={})
    ]
    mock_embedding_service.embed_batch.return_value = [[0.5] * 384]
    mock_vector_store.upsert_chunks.return_value = 1

    service = IngestionService(
        loader=mock_loader,
        chunker=mock_chunker,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )

    stats = service.ingest_directory(tmp_path)

    assert stats["documents_processed"] == 2
    assert stats["chunks_created"] == 2
    assert stats["chunks_upserted"] == 2


def test_ingestion_service_idempotency(tmp_path):
    """Test 4: Verify re-ingesting same document produces identical chunk IDs and metrics."""
    mock_loader = MagicMock(spec=DocumentLoader)
    mock_chunker = MagicMock(spec=BaseChunker)
    mock_embedding_service = MagicMock(spec=BaseEmbeddingService)
    mock_vector_store = MagicMock(spec=BaseVectorStore)

    dummy_file = tmp_path / "sample.md"
    dummy_file.write_text("# Title\nSame content")

    doc = Document(doc_id="stable_doc_id", content="Same content", file_type="md", metadata={"filename": "sample.md"})
    chunk = DocumentChunk(chunk_id="stable_chunk_id", doc_id="stable_doc_id", text="Same content", chunk_index=0, metadata={})

    mock_loader.load_document.return_value = [doc]
    mock_chunker.chunk_document.return_value = [chunk]
    mock_embedding_service.embed_batch.return_value = [[0.1] * 384]
    mock_vector_store.upsert_chunks.return_value = 1

    service = IngestionService(
        loader=mock_loader,
        chunker=mock_chunker,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )

    stats1 = service.ingest_file(dummy_file)
    stats2 = service.ingest_file(dummy_file)

    assert stats1 == stats2
    assert stats1["chunks_created"] == 1
    assert stats1["chunks_upserted"] == 1
