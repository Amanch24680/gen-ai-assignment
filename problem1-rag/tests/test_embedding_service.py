import math
import pytest
from app.embeddings.service import SentenceTransformerEmbeddingService


def test_embedding_service_unit_defaults():
    """Unit test: Verify default model initialization and dimension property."""
    service = SentenceTransformerEmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    assert service.model_name == "BAAI/bge-small-en-v1.5"
    assert service.dimension == 384
    assert service._model is None  # Lazy loading: model not loaded yet


def test_embedding_service_empty_input_handling():
    """Unit test: Verify empty text returns zero vector and empty batch returns []."""
    service = SentenceTransformerEmbeddingService(model_name="BAAI/bge-small-en-v1.5")

    # Empty string
    zero_vec = service.embed_text("")
    assert len(zero_vec) == 384
    assert all(v == 0.0 for v in zero_vec)

    # Empty batch
    assert service.embed_batch([]) == []


@pytest.mark.integration
def test_embedding_service_real_model_inference():
    """
    Integration test: Load real BAAI/bge-small-en-v1.5 model and verify:
    - Embedding succeeds
    - Dimension == 384
    - Single text output shape
    - Batch output shape
    - L2 normalization (L2 norm ~ 1.0)
    - Ordering preservation
    """
    service = SentenceTransformerEmbeddingService(model_name="BAAI/bge-small-en-v1.5")

    # 1. Single text embedding
    text = "RAG application embedding test sentence."
    vec = service.embed_text(text)

    assert len(vec) == 384
    # Verify L2 norm == 1.0 (normalized vector)
    l2_norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(l2_norm, 1.0, abs_tol=1e-3)

    # 2. Batch embedding & ordering preservation
    texts = [
        "First sentence for batch embedding.",
        "Second completely different sentence.",
        "Third sentence for verification.",
    ]
    batch_vecs = service.embed_batch(texts)

    assert len(batch_vecs) == 3
    for v in batch_vecs:
        assert len(v) == 384

    # Verify order is preserved: embed individually and compare vectors
    vec_0_single = service.embed_text(texts[0])
    vec_2_single = service.embed_text(texts[2])

    assert pytest.approx(batch_vecs[0], abs=1e-4) == vec_0_single
    assert pytest.approx(batch_vecs[2], abs=1e-4) == vec_2_single
