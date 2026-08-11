class RetrievalError(Exception):
    """Base exception for retrieval operation failures."""
    pass


class EmptyQueryError(RetrievalError, ValueError):
    """Raised when query input is empty or whitespace-only."""
    pass


class InvalidRetrievalConfigError(RetrievalError, ValueError):
    """Raised when top_k or relevance_threshold parameters are out of valid range."""
    pass
