"""
Evaluation package for Problem 1 RAG pipeline.
"""
from pathlib import Path
import json
from typing import Any, Dict, List

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


def load_evaluation_dataset() -> List[Dict[str, Any]]:
    """Load and return the fixed evaluation dataset JSON."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
