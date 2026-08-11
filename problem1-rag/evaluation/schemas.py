from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationQuestionResult(BaseModel):
    """Structured evaluation result for a single question case."""
    question_id: str = Field(description="Unique question identifier")
    question: str = Field(description="User evaluation question text")
    category: str = Field(description="Category of query (direct_fact, multi_chunk, unanswerable, etc.)")
    relevant_documents: List[str] = Field(default_factory=list, description="Ground truth document IDs")
    relevant_chunk_ids: List[str] = Field(default_factory=list, description="Ground truth chunk IDs")
    retrieved_chunk_ids: List[str] = Field(default_factory=list, description="Retrieved chunk IDs")
    retrieved_scores: List[float] = Field(default_factory=list, description="Retrieved similarity scores")
    recall_at_1: float = Field(default=0.0, description="Recall@1 metric")
    recall_at_3: float = Field(default=0.0, description="Recall@3 metric")
    recall_at_5: float = Field(default=0.0, description="Recall@5 metric")
    precision_at_1: float = Field(default=0.0, description="Precision@1 metric")
    precision_at_3: float = Field(default=0.0, description="Precision@3 metric")
    precision_at_5: float = Field(default=0.0, description="Precision@5 metric")
    reciprocal_rank: float = Field(default=0.0, description="Reciprocal Rank metric")
    is_unanswerable: bool = Field(default=False, description="Flag indicating out-of-corpus question")
    retrieval_returned_results: bool = Field(default=False, description="Flag indicating whether retrieval returned any results")


class RetrievalEvaluationSummary(BaseModel):
    """Aggregate evaluation summary across all dataset questions."""
    total_cases: int = Field(description="Total number of evaluated cases")
    answerable_cases: int = Field(description="Count of answerable cases")
    unanswerable_cases: int = Field(description="Count of unanswerable cases")
    recall_at_1: float = Field(description="Mean Recall@1 across answerable cases")
    recall_at_3: float = Field(description="Mean Recall@3 across answerable cases")
    recall_at_5: float = Field(description="Mean Recall@5 across answerable cases")
    precision_at_1: float = Field(description="Mean Precision@1 across answerable cases")
    precision_at_3: float = Field(description="Mean Precision@3 across answerable cases")
    precision_at_5: float = Field(description="Mean Precision@5 across answerable cases")
    mrr: float = Field(description="Mean Reciprocal Rank (MRR) across answerable cases")
    unanswerable_empty_retrieval_count: int = Field(description="Count of unanswerable cases that returned 0 results")
    unanswerable_non_empty_retrieval_count: int = Field(description="Count of unanswerable cases that returned >0 results")
    unanswerable_empty_retrieval_rate: float = Field(description="Rate of empty retrievals for unanswerable cases")
    category_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Category-level metrics breakdown")
    per_question_results: List[EvaluationQuestionResult] = Field(default_factory=list, description="Per-question evaluation results")
