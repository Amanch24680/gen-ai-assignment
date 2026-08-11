import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.embeddings.base import BaseEmbeddingService
from app.embeddings.service import SentenceTransformerEmbeddingService
from app.ingestion.base import BaseChunker
from app.ingestion.chunker import TextChunker
from app.ingestion.parsers import DocumentLoader
from app.schemas.document import Document, DocumentChunk
from app.vector_store.base import BaseVectorStore
from app.vector_store.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown"}


class IngestionService:
    """
    Orchestration service for the document ingestion pipeline.
    Combines document loading/parsing, text chunking, embedding generation,
    and vector store persistence.
    """

    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[BaseChunker] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or TextChunker()
        self.embedding_service = embedding_service or SentenceTransformerEmbeddingService()
        self.vector_store = vector_store or QdrantVectorStore()

    def ingest_file(self, file_path: Union[str, Path]) -> Dict[str, int]:
        """
        Ingest a single document file.
        1. Load & parse document into Document objects.
        2. Chunk Document objects into DocumentChunk objects.
        3. Generate batch embeddings and attach to chunks.
        4. Upsert chunks into vector store.
        Returns dictionary of ingestion statistics.
        """
        path = Path(file_path).resolve()
        logger.info(f"Ingesting file: {path.name}")

        documents: List[Document] = self.loader.load_document(path)
        all_chunks: List[DocumentChunk] = []

        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.info(f"No chunks generated for file '{path.name}'.")
            return {
                "documents_processed": len(documents),
                "chunks_created": 0,
                "chunks_upserted": 0,
            }

        # Generate embeddings in batch
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedding_service.embed_batch(texts)

        # Attach embeddings to chunks
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        # Upsert chunks into vector store
        upsert_count = self.vector_store.upsert_chunks(all_chunks)
        logger.info(
            f"Successfully ingested '{path.name}': {len(documents)} docs, "
            f"{len(all_chunks)} chunks created, {upsert_count} chunks upserted."
        )

        return {
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks),
            "chunks_upserted": upsert_count,
        }

    def ingest_directory(
        self, dir_path: Union[str, Path], recursive: bool = True
    ) -> Dict[str, int]:
        """
        Ingest all supported document files in a directory.
        Returns aggregated ingestion statistics.
        """
        path = Path(dir_path).resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Directory path does not exist or is not a directory: {path}")

        pattern = "**/*" if recursive else "*"
        files_to_process = sorted([
            p for p in path.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        logger.info(f"Found {len(files_to_process)} supported files in directory '{path}'")

        total_docs = 0
        total_chunks = 0
        total_upserted = 0

        for file_p in files_to_process:
            try:
                stats = self.ingest_file(file_p)
                total_docs += stats["documents_processed"]
                total_chunks += stats["chunks_created"]
                total_upserted += stats["chunks_upserted"]
            except Exception as exc:
                logger.error(f"Failed to ingest file '{file_p.name}': {exc}")

        return {
            "documents_processed": total_docs,
            "chunks_created": total_chunks,
            "chunks_upserted": total_upserted,
        }
