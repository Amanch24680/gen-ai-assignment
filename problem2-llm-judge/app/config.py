from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for Problem 2 LLM-as-Judge Evaluation Pipeline."""

    # Generator and Judge models are independently configurable
    generator_model: str = "gemma:2b"
    judge_model: str = "llama3:latest"

    # Ollama Service Settings
    ollama_base_url: str = "http://localhost:11434"
    judge_temperature: float = 0.0
    judge_timeout: float = 60.0

    # Evaluation Thresholds
    pass_score_threshold: float = 3.5

    # Directory Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    results_dir: Path = Path(__file__).resolve().parent.parent / "results"
    datasets_dir: Path = Path(__file__).resolve().parent.parent / "datasets"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
