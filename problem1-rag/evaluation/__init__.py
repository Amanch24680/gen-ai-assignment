"""
Evaluation package for Problem 1 RAG pipeline.
"""
from evaluation.loader import DATASET_PATH, load_evaluation_dataset
from evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.schemas import (
    EvaluationQuestionResult,
    RetrievalEvaluationSummary,
)

__all__ = [
    "DATASET_PATH",
    "load_evaluation_dataset",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "EvaluationQuestionResult",
    "RetrievalEvaluationSummary",
    "RetrievalEvaluator",
]
