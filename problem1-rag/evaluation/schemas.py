from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationQuestionResult(BaseModel):
    """Structured evaluation result for a single retrieval evaluation case."""
    question_id: str = Field(description="Unique question identifier")
    question: str = Field(description="User evaluation question text")
    category: str = Field(description="Category of query")
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
    """Aggregate evaluation summary across all dataset questions for retrieval."""
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


class RAGAnswerQuestionResult(BaseModel):
    """Structured end-to-end evaluation result for a single RAG answer case."""
    question_id: str = Field(description="Unique question identifier")
    question: str = Field(description="User question text")
    category: str = Field(description="Category of query")
    ground_truth_answer: Optional[str] = Field(default=None, description="Ground truth reference answer")
    generated_answer: str = Field(default="", description="LLM generated answer text")
    retrieved_chunk_ids: List[str] = Field(default_factory=list, description="Retrieved chunk IDs")
    cited_chunk_ids: List[str] = Field(default_factory=list, description="Cited chunk IDs")
    has_relevant_context: bool = Field(default=False, description="Flag indicating relevant context was found")
    latency_ms: float = Field(default=0.0, description="Execution latency in milliseconds")
    answer_f1: float = Field(default=0.0, description="Token-level answer F1 score vs ground truth")
    context_support: float = Field(default=0.0, description="Lexical context support score")
    citation_coverage: float = Field(default=0.0, description="Citation coverage score")
    is_unanswerable: bool = Field(default=False, description="Flag indicating unanswerable question")
    unanswerable_safe_handled: bool = Field(default=False, description="Flag indicating safe handling of unanswerable question")
    success: bool = Field(default=True, description="Flag indicating generation succeeded without error")
    error_message: Optional[str] = Field(default=None, description="Error message if generation failed")


class RAGAnswerEvaluationSummary(BaseModel):
    """Aggregate evaluation summary across all dataset questions for end-to-end RAG answer pipeline."""
    total_cases: int = Field(description="Total number of evaluated cases")
    answerable_cases: int = Field(description="Count of answerable cases")
    unanswerable_cases: int = Field(description="Count of unanswerable cases")
    successful_generations: int = Field(description="Count of successful generations")
    failed_generations: int = Field(description="Count of failed generations")
    mean_answer_f1: float = Field(description="Mean answer token F1 score across answerable cases")
    mean_context_support: float = Field(description="Mean lexical context support score across answerable cases")
    mean_citation_coverage: float = Field(description="Mean citation coverage score across answerable cases")
    unanswerable_safe_handling_rate: float = Field(description="Rate of safe handling for unanswerable questions")
    average_latency_ms: float = Field(description="Average execution latency in milliseconds")
    median_latency_ms: float = Field(description="Median execution latency in milliseconds")
    min_latency_ms: float = Field(description="Minimum execution latency in milliseconds")
    max_latency_ms: float = Field(description="Maximum execution latency in milliseconds")
    category_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Category-level metrics breakdown")
    per_question_results: List[RAGAnswerQuestionResult] = Field(default_factory=list, description="Per-question evaluation results")
