from typing import Self
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration loaded via Pydantic Settings.
    Supports environment variable overrides and provides fallback defaults.
    """

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace embedding model name"
    )
    vector_store_path: str = Field(
        default="./data/qdrant",
        description="Path to local Qdrant vector store database directory"
    )
    chunk_size: int = Field(
        default=800,
        description="Target document chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=120,
        description="Overlap between consecutive document chunks in characters"
    )
    top_k: int = Field(
        default=5,
        description="Number of top candidate chunks to retrieve"
    )
    relevance_threshold: float = Field(
        default=0.35,
        description="Similarity threshold below which retrieved chunks are filtered out"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama HTTP API service"
    )
    generator_model: str = Field(
        default="gemma:2b",
        description="LLM generator model name available in Ollama"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("CHUNK_SIZE must be strictly positive (> 0).")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative (>= 0).")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("TOP_K must be strictly positive (> 0).")
        return v

    @field_validator("relevance_threshold")
    @classmethod
    def validate_relevance_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("RELEVANCE_THRESHOLD must be between 0.0 and 1.0 inclusive.")
        return v

    @model_validator(mode="after")
    def validate_overlap_less_than_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be strictly less than CHUNK_SIZE.")
        return self


def get_settings(**kwargs) -> Settings:
    """
    Retrieve application settings instance, accepting optional keyword parameter overrides.
    """
    return Settings(**kwargs)
