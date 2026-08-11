"""
Custom exceptions for document ingestion and chunking pipeline.
"""


class IngestionError(Exception):
    """Base exception for all document ingestion and chunking errors."""
    pass


class UnsupportedFileTypeError(IngestionError):
    """Raised when an unsupported file extension is provided."""
    pass


class DocumentParsingError(IngestionError):
    """Raised when a document cannot be opened, read, or parsed."""
    pass


class EmptyDocumentError(IngestionError):
    """Raised when an ingested document contains no extractable text."""
    pass


class InvalidChunkConfigError(IngestionError):
    """Raised when chunk size or overlap parameters are invalid."""
    pass
