import logging
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings
from app.embeddings.base import BaseEmbeddingService

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """
    Embedding service utilizing sentence-transformers (BAAI/bge-small-en-v1.5 by default).
    Loads the underlying SentenceTransformer model lazily upon first invocation.
    Caches model instances across invocations to preserve PyTorch process stability.
    Employs normalized L2 embeddings suitable for Cosine distance vector search.
    """

    _model_cache: Dict[str, SentenceTransformer] = {}

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name if model_name is not None else settings.embedding_model
        self._model: Optional[SentenceTransformer] = None
        self._dimension: int = 384

    def _get_model(self) -> SentenceTransformer:
        """Lazy load and cache the SentenceTransformer model instance."""
        if self._model is None:
            if self.model_name not in SentenceTransformerEmbeddingService._model_cache:
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                SentenceTransformerEmbeddingService._model_cache[self.model_name] = SentenceTransformer(self.model_name)
            self._model = SentenceTransformerEmbeddingService._model_cache[self.model_name]
            try:
                self._dimension = self._model.get_embedding_dimension()
            except AttributeError:
                try:
                    self._dimension = self._model.get_sentence_embedding_dimension()
                except AttributeError:
                    self._dimension = 384
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        if self._model is not None:
            return self._dimension
        return 384

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string into a normalized 384-dimensional float vector.
        Empty or whitespace-only inputs return a zero vector of dimension 384.
        """
        if not text or not text.strip():
            return [0.0] * self.dimension

        model = self._get_model()
        # normalize_embeddings=True normalizes vectors to L2 norm = 1.0, ideal for Cosine distance
        vector = model.encode(text, normalize_embeddings=True).tolist()
        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings into a list of normalized vector lists.
        Preserves input list ordering. Empty inputs return [].
        Empty or whitespace-only elements return zero vectors.
        """
        if not texts:
            return []

        # Track indices of non-empty text strings
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        results: List[List[float]] = [[0.0] * self.dimension for _ in range(len(texts))]

        if non_empty_texts:
            model = self._get_model()
            vectors = model.encode(non_empty_texts, normalize_embeddings=True).tolist()
            for idx, vec in zip(non_empty_indices, vectors):
                results[idx] = vec

        return results
