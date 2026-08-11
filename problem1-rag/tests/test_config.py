import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.config.settings import Settings as SettingsFromModule


def test_settings_importable():
    """Verify that Settings and get_settings are importable from app.config package."""
    assert Settings is not None
    assert get_settings is not None
    assert Settings is SettingsFromModule


def test_default_settings_load_correctly(monkeypatch):
    """Verify default configuration settings load with expected default values when no env overrides exist."""
    # Ensure env vars are clear for test isolation
    for env_var in [
        "EMBEDDING_MODEL",
        "VECTOR_STORE_PATH",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "RELEVANCE_THRESHOLD",
        "OLLAMA_BASE_URL",
        "GENERATOR_MODEL",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    settings = get_settings(_env_file=None)

    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.vector_store_path == "./data/qdrant"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.top_k == 5
    assert settings.relevance_threshold == 0.35
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.generator_model == "gemma:2b"


def test_environment_variables_override_defaults(monkeypatch):
    """Verify environment variables override default settings values."""
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/embedding-model")
    monkeypatch.setenv("VECTOR_STORE_PATH", "/custom/path/qdrant")
    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    monkeypatch.setenv("TOP_K", "10")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.5")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-host:11434")
    monkeypatch.setenv("GENERATOR_MODEL", "custom-gemma:2b")

    settings = Settings(_env_file=None)

    assert settings.embedding_model == "custom/embedding-model"
    assert settings.vector_store_path == "/custom/path/qdrant"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.top_k == 10
    assert settings.relevance_threshold == 0.5
    assert settings.ollama_base_url == "http://ollama-host:11434"
    assert settings.generator_model == "custom-gemma:2b"


@pytest.mark.parametrize("invalid_size", [0, -100])
def test_invalid_chunk_size_rejected(invalid_size):
    """Verify invalid non-positive chunk size is rejected by validation."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(chunk_size=invalid_size, _env_file=None)
    assert "CHUNK_SIZE must be strictly positive" in str(exc_info.value)


@pytest.mark.parametrize("invalid_overlap", [-1, -50])
def test_invalid_negative_chunk_overlap_rejected(invalid_overlap):
    """Verify negative chunk overlap is rejected by validation."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(chunk_overlap=invalid_overlap, _env_file=None)
    assert "CHUNK_OVERLAP must be non-negative" in str(exc_info.value)


@pytest.mark.parametrize(
    "size,overlap",
    [(500, 500), (500, 600)],
)
def test_chunk_overlap_greater_or_equal_to_size_rejected(size, overlap):
    """Verify chunk overlap greater than or equal to chunk size is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(chunk_size=size, chunk_overlap=overlap, _env_file=None)
    assert "CHUNK_OVERLAP must be strictly less than CHUNK_SIZE" in str(exc_info.value)


@pytest.mark.parametrize("invalid_top_k", [0, -5])
def test_invalid_top_k_rejected(invalid_top_k):
    """Verify non-positive top_k is rejected by validation."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(top_k=invalid_top_k, _env_file=None)
    assert "TOP_K must be strictly positive" in str(exc_info.value)


@pytest.mark.parametrize("invalid_threshold", [-0.1, 1.5, 2.0])
def test_invalid_relevance_threshold_rejected(invalid_threshold):
    """Verify relevance threshold outside [0.0, 1.0] range is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(relevance_threshold=invalid_threshold, _env_file=None)
    assert "RELEVANCE_THRESHOLD must be between 0.0 and 1.0" in str(exc_info.value)
