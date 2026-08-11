"""
Evaluation package for Problem 1 RAG pipeline.
"""
from evaluation.loader import DATASET_PATH, load_evaluation_dataset
from evaluation.metrics import (
    answer_token_f1,
    citation_coverage_score,
    context_support_score,
    mean_reciprocal_rank,
    normalize_text,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unanswerable_safe_handling,
)
from evaluation.rag_evaluator import RAGEvaluator
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.schemas import (
    EvaluationQuestionResult,
    RAGAnswerEvaluationSummary,
    RAGAnswerQuestionResult,
    RetrievalEvaluationSummary,
)

__all__ = [
    "DATASET_PATH",
    "load_evaluation_dataset",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "normalize_text",
    "answer_token_f1",
    "context_support_score",
    "citation_coverage_score",
    "unanswerable_safe_handling",
    "EvaluationQuestionResult",
    "RetrievalEvaluationSummary",
    "RAGAnswerQuestionResult",
    "RAGAnswerEvaluationSummary",
    "RetrievalEvaluator",
    "RAGEvaluator",
]
