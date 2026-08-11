class GenerationError(Exception):
    """Base exception for LLM generation failures."""
    pass


class OllamaConnectionError(GenerationError):
    """Raised when connection to Ollama HTTP API fails."""
    pass


class OllamaTimeoutError(GenerationError):
    """Raised when request to Ollama HTTP API times out."""
    pass


class OllamaResponseError(GenerationError):
    """Raised when Ollama returns an HTTP error, malformed JSON, or unexpected payload."""
    pass


class EmptyGenerationError(GenerationError):
    """Raised when Ollama returns an empty or whitespace-only response string."""
    pass
