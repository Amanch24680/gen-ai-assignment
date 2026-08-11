from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingService(ABC):
    """Abstract base class for vector embedding generation services."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a vector float representation."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings into vector float representations."""
        pass
