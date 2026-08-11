import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config.settings import get_settings
from app.schemas.document import DocumentChunk
from app.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)


def chunk_id_to_qdrant_id(chunk_id: str) -> str:
    """
    Deterministically convert any chunk_id string into a valid Qdrant UUID point ID string.
    If chunk_id is a valid 32-character hex string, converts directly.
    Otherwise, computes SHA-256 hash hex string to guarantee valid UUID conversion.
    """
    if len(chunk_id) == 32:
        try:
            return str(uuid.UUID(hex=chunk_id))
        except ValueError:
            pass
    hex_str = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:32]
    return str(uuid.UUID(hex=hex_str))


class QdrantVectorStore(BaseVectorStore):
    """
    Local persistent Qdrant vector store implementation.
    Operates on local storage directory specified in VECTOR_STORE_PATH or in-memory for testing.
    Uses Cosine distance metric and stores rich payload metadata alongside vectors.
    """

    def __init__(
        self,
        collection_name: str = "document_chunks",
        vector_size: int = 384,
        path: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ):
        settings = get_settings()
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.storage_path = path if path is not None else settings.vector_store_path

        if client is not None:
            self.client = client
        elif self.storage_path == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            path_obj = Path(self.storage_path).resolve()
            path_obj.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path_obj))

        self.create_collection_if_not_exists()

    def collection_exists(self) -> bool:
        """Check if the collection exists in Qdrant."""
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception:
            return False

    def create_collection_if_not_exists(self) -> None:
        """Create the collection with specified vector_size and Cosine distance if it doesn't exist."""
        if not self.collection_exists():
            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' with vector_size={self.vector_size}, distance=COSINE"
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Upsert a list of DocumentChunks into Qdrant.
        Each chunk must have chunk.embedding populated.
        Point ID is derived deterministically from chunk.chunk_id using UUID.
        Payload metadata includes original chunk_id, doc_id, source, filename, file_type, page_number, chunk_index, text.
        Returns the number of points upserted.
        """
        if not chunks:
            return 0

        points: List[models.PointStruct] = []
        for chunk in chunks:
            if chunk.embedding is None or len(chunk.embedding) == 0:
                raise ValueError(f"Chunk '{chunk.chunk_id}' has no embedding vector populated.")

            point_id = chunk_id_to_qdrant_id(chunk.chunk_id)

            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "document_id": chunk.doc_id,
                "source": chunk.metadata.get("source", ""),
                "filename": chunk.metadata.get("filename", ""),
                "file_type": chunk.metadata.get("file_type", ""),
                "page_number": chunk.metadata.get("page_number"),
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            }
            # Copy any additional metadata key-value pairs
            for k, v in chunk.metadata.items():
                if k not in payload:
                    payload[k] = v

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    def count_chunks(self) -> int:
        """Return total count of chunks in the collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        relevance_threshold: float,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Low-level search primitive. Finds top_k points matching query_vector and optional metadata filter.
        Applies relevance_threshold filtering on similarity score.
        """
        qdrant_filter: Optional[models.Filter] = None
        if metadata_filter:
            must_conditions = []
            for key, val in metadata_filter.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=val),
                    )
                )
            qdrant_filter = models.Filter(must=must_conditions)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=relevance_threshold,
        )

        results: List[DocumentChunk] = []
        for point in search_result.points:
            p = point.payload or {}
            score = point.score
            meta = dict(p)
            meta["score"] = score
            chunk = DocumentChunk(
                chunk_id=p.get("chunk_id", str(point.id)),
                doc_id=p.get("doc_id", p.get("document_id", "")),
                text=p.get("text", ""),
                chunk_index=p.get("chunk_index", 0),
                metadata=meta,
                embedding=point.vector if isinstance(point.vector, list) else None,
                score=score,
            )
            results.append(chunk)

        return results

    def reset_collection(self) -> None:
        """Reset and clear all points from the collection (useful for test teardown)."""
        if self.collection_exists():
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.Filter(),
                )
            except Exception:
                self.client.delete_collection(self.collection_name)
                self.create_collection_if_not_exists()

    def close(self) -> None:
        """Close the local Qdrant client connection."""
        try:
            self.client.close()
        except Exception:
            pass
