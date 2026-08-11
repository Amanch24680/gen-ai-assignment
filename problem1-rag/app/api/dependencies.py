from functools import lru_cache
from app.embeddings.service import SentenceTransformerEmbeddingService
from app.generation.service import OllamaGenerator
from app.rag.service import RAGService
from app.retrieval.service import VectorRetriever
from app.vector_store.qdrant_store import QdrantVectorStore


@lru_cache()
def get_embedding_service() -> SentenceTransformerEmbeddingService:
    """Singleton provider for SentenceTransformerEmbeddingService."""
    return SentenceTransformerEmbeddingService()


@lru_cache()
def get_vector_store() -> QdrantVectorStore:
    """Singleton provider for QdrantVectorStore."""
    return QdrantVectorStore()


@lru_cache()
def get_retriever() -> VectorRetriever:
    """Singleton provider for VectorRetriever."""
    return VectorRetriever(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@lru_cache()
def get_generator() -> OllamaGenerator:
    """Singleton provider for OllamaGenerator."""
    return OllamaGenerator()


@lru_cache()
def get_rag_service() -> RAGService:
    """Singleton provider for RAGService orchestration layer."""
    return RAGService(
        retriever=get_retriever(),
        generator=get_generator(),
    )
