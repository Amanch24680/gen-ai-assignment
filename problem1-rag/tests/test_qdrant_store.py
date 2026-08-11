import uuid
import pytest
from app.schemas.document import DocumentChunk
from app.vector_store.qdrant_store import QdrantVectorStore, chunk_id_to_qdrant_id


def test_chunk_id_to_qdrant_id_deterministic():
    """Verify chunk_id_to_qdrant_id produces valid and deterministic UUID string."""
    chunk_id = "9a2caeacc9a957819ffd162db17f0123"
    q_id_1 = chunk_id_to_qdrant_id(chunk_id)
    q_id_2 = chunk_id_to_qdrant_id(chunk_id)

    assert q_id_1 == q_id_2
    # Verify it is a valid UUID string format
    parsed_uuid = uuid.UUID(q_id_1)
    assert str(parsed_uuid) == q_id_1


def test_qdrant_store_in_memory_initialization_and_collection(tmp_path):
    """Test QdrantVectorStore local initialization, collection creation, and dimension configuration."""
    store = QdrantVectorStore(
        collection_name="test_collection",
        vector_size=384,
        path=":memory:",
    )

    assert store.collection_exists()
    assert store.count_chunks() == 0
    store.close()


def test_qdrant_store_upsert_payload_and_idempotency(tmp_path):
    """Test chunk upsert, metadata payload retention, and repeated upsert idempotency."""
    db_path = str(tmp_path / "qdrant_test_db")
    store = QdrantVectorStore(
        collection_name="test_chunks",
        vector_size=384,
        path=db_path,
    )

    # Use distinct orthogonal vectors so cosine similarity cleanly ranks chunk_1 top for vec_1 query
    vec_1 = [1.0] + [0.0] * 383
    vec_2 = [0.0, 1.0] + [0.0] * 382

    chunk_1 = DocumentChunk(
        chunk_id="9a2caeacc9a957819ffd162db17f0123",
        doc_id="doc_100",
        text="Sample vector store payload chunk text.",
        chunk_index=0,
        metadata={
            "filename": "sample.pdf",
            "source": "/path/to/sample.pdf",
            "file_type": "pdf",
            "page_number": 1,
        },
        embedding=vec_1,
    )

    chunk_2 = DocumentChunk(
        chunk_id="5a17606523cd4810e3b69626903c4567",
        doc_id="doc_100",
        text="Second chunk text.",
        chunk_index=1,
        metadata={
            "filename": "sample.pdf",
            "source": "/path/to/sample.pdf",
            "file_type": "pdf",
            "page_number": 1,
        },
        embedding=vec_2,
    )

    # 1. Upsert chunks
    upsert_count = store.upsert_chunks([chunk_1, chunk_2])
    assert upsert_count == 2
    assert store.count_chunks() == 2

    # 2. Idempotency test: Upsert same chunks again
    upsert_count_again = store.upsert_chunks([chunk_1, chunk_2])
    assert upsert_count_again == 2
    assert store.count_chunks() == 2  # Count must remain 2 (no duplicates created)

    # 3. Verify search & payload retrieval
    results = store.search(
        query_vector=vec_1,
        top_k=2,
        relevance_threshold=0.0,
    )
    assert len(results) == 2
    top_chunk = results[0]
    assert top_chunk.chunk_id == chunk_1.chunk_id
    assert top_chunk.doc_id == "doc_100"
    assert top_chunk.text == chunk_1.text
    assert top_chunk.metadata["filename"] == "sample.pdf"
    assert top_chunk.metadata["page_number"] == 1

    # 4. Test reset_collection
    store.reset_collection()
    assert store.count_chunks() == 0

    store.close()


def test_qdrant_store_persistence_and_reopen(tmp_path):
    """
    Test persistence across closing and reopening local Qdrant store:
    - Upsert data into local store
    - Close client connection
    - Re-instantiate QdrantVectorStore at same path
    - Verify collection and data persist intact
    """
    db_path = str(tmp_path / "persistent_qdrant")

    # Step 1: Open store and upsert
    store_1 = QdrantVectorStore(
        collection_name="persistent_chunks",
        vector_size=384,
        path=db_path,
    )

    chunk = DocumentChunk(
        chunk_id="11112222333344445555666677778888",
        doc_id="doc_persist",
        text="Persistent chunk data surviving restart.",
        chunk_index=0,
        metadata={"filename": "persist.md", "source": "/path/persist.md", "file_type": "md"},
        embedding=[1.0] + [0.0] * 383,
    )

    store_1.upsert_chunks([chunk])
    assert store_1.count_chunks() == 1

    # Step 2: Close connection
    store_1.close()

    # Step 3: Reopen connection at same storage directory
    store_2 = QdrantVectorStore(
        collection_name="persistent_chunks",
        vector_size=384,
        path=db_path,
    )

    assert store_2.collection_exists()
    assert store_2.count_chunks() == 1

    results = store_2.search(
        query_vector=[1.0] + [0.0] * 383,
        top_k=1,
        relevance_threshold=0.0,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "11112222333344445555666677778888"
    assert results[0].text == "Persistent chunk data surviving restart."
    assert results[0].metadata["filename"] == "persist.md"

    store_2.close()
