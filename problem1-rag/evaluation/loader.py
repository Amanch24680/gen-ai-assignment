import json
from pathlib import Path
from typing import Any, Dict, List

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


def load_evaluation_dataset(dataset_path: Path = None) -> List[Dict[str, Any]]:
    """Load and return the fixed evaluation dataset JSON."""
    target_path = dataset_path or DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Evaluation dataset file not found: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)
