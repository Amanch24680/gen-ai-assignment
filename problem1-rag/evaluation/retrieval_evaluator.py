import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.retrieval.base import BaseRetriever
from app.retrieval.service import VectorRetriever
from evaluation.loader import load_evaluation_dataset
from evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.schemas import EvaluationQuestionResult, RetrievalEvaluationSummary

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Deterministic retrieval pipeline evaluator.
    Evaluates retrieval quality against ground truth dataset without calling LLM generation.
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        dataset_path: Optional[Path] = None,
    ):
        self.retriever = retriever or VectorRetriever()
        self.dataset_path = dataset_path

    def evaluate(
        self,
        top_k: int = 5,
        relevance_threshold: Optional[float] = None,
    ) -> RetrievalEvaluationSummary:
        """
        Run retrieval evaluation across all cases in the dataset.
        Isolates the retrieval layer (no LLM generation called).
        """
        dataset = load_evaluation_dataset(self.dataset_path)
        logger.info(f"Loaded {len(dataset)} evaluation cases for retrieval evaluation.")

        per_question_results: List[EvaluationQuestionResult] = []

        answerable_cases: List[EvaluationQuestionResult] = []
        unanswerable_cases: List[EvaluationQuestionResult] = []

        for case in dataset:
            q_id = case["id"]
            question = case["question"]
            category = case["category"]
            rel_docs = case.get("relevant_documents", [])
            rel_chunks = case.get("relevant_chunk_ids", [])

            # Invoke retrieval service directly (no generator call)
            retrieved_chunks = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                relevance_threshold=relevance_threshold,
            )

            retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
            retrieved_scores = [
                c.score if c.score is not None else c.metadata.get("score", 0.0)
                for c in retrieved_chunks
            ]

            if category == "unanswerable":
                result = EvaluationQuestionResult(
                    question_id=q_id,
                    question=question,
                    category=category,
                    relevant_documents=rel_docs,
                    relevant_chunk_ids=rel_chunks,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieved_scores=retrieved_scores,
                    recall_at_1=0.0,
                    recall_at_3=0.0,
                    recall_at_5=0.0,
                    precision_at_1=0.0,
                    precision_at_3=0.0,
                    precision_at_5=0.0,
                    reciprocal_rank=0.0,
                    is_unanswerable=True,
                    retrieval_returned_results=(len(retrieved_chunk_ids) > 0),
                )
                unanswerable_cases.append(result)
            else:
                r1 = recall_at_k(retrieved_chunk_ids, rel_chunks, 1)
                r3 = recall_at_k(retrieved_chunk_ids, rel_chunks, 3)
                r5 = recall_at_k(retrieved_chunk_ids, rel_chunks, 5)

                p1 = precision_at_k(retrieved_chunk_ids, rel_chunks, 1)
                p3 = precision_at_k(retrieved_chunk_ids, rel_chunks, 3)
                p5 = precision_at_k(retrieved_chunk_ids, rel_chunks, 5)

                rr = reciprocal_rank(retrieved_chunk_ids, rel_chunks)

                result = EvaluationQuestionResult(
                    question_id=q_id,
                    question=question,
                    category=category,
                    relevant_documents=rel_docs,
                    relevant_chunk_ids=rel_chunks,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieved_scores=retrieved_scores,
                    recall_at_1=r1,
                    recall_at_3=r3,
                    recall_at_5=r5,
                    precision_at_1=p1,
                    precision_at_3=p3,
                    precision_at_5=p5,
                    reciprocal_rank=rr,
                    is_unanswerable=False,
                    retrieval_returned_results=(len(retrieved_chunk_ids) > 0),
                )
                answerable_cases.append(result)

            per_question_results.append(result)

        # Aggregate answerable metrics
        n_ans = len(answerable_cases)
        if n_ans > 0:
            mean_r1 = sum(c.recall_at_1 for c in answerable_cases) / n_ans
            mean_r3 = sum(c.recall_at_3 for c in answerable_cases) / n_ans
            mean_r5 = sum(c.recall_at_5 for c in answerable_cases) / n_ans

            mean_p1 = sum(c.precision_at_1 for c in answerable_cases) / n_ans
            mean_p3 = sum(c.precision_at_3 for c in answerable_cases) / n_ans
            mean_p5 = sum(c.precision_at_5 for c in answerable_cases) / n_ans

            mrr = mean_reciprocal_rank([c.reciprocal_rank for c in answerable_cases])
        else:
            mean_r1 = mean_r3 = mean_r5 = 0.0
            mean_p1 = mean_p3 = mean_p5 = 0.0
            mrr = 0.0

        # Unanswerable statistics
        n_unans = len(unanswerable_cases)
        empty_unans_count = sum(1 for c in unanswerable_cases if not c.retrieval_returned_results)
        non_empty_unans_count = n_unans - empty_unans_count
        empty_unans_rate = (empty_unans_count / n_unans) if n_unans > 0 else 0.0

        # Category breakdowns
        category_groups: Dict[str, List[EvaluationQuestionResult]] = {}
        for res in per_question_results:
            category_groups.setdefault(res.category, []).append(res)

        category_metrics: Dict[str, Dict[str, float]] = {}
        for cat, items in category_groups.items():
            if cat == "unanswerable":
                cat_empty = sum(1 for item in items if not item.retrieval_returned_results)
                category_metrics[cat] = {
                    "count": float(len(items)),
                    "empty_retrieval_rate": (cat_empty / len(items)) if items else 0.0,
                }
            else:
                cat_n = len(items)
                category_metrics[cat] = {
                    "count": float(cat_n),
                    "recall_at_1": sum(i.recall_at_1 for i in items) / cat_n,
                    "recall_at_3": sum(i.recall_at_3 for i in items) / cat_n,
                    "recall_at_5": sum(i.recall_at_5 for i in items) / cat_n,
                    "precision_at_1": sum(i.precision_at_1 for i in items) / cat_n,
                    "precision_at_3": sum(i.precision_at_3 for i in items) / cat_n,
                    "precision_at_5": sum(i.precision_at_5 for i in items) / cat_n,
                    "mrr": sum(i.reciprocal_rank for i in items) / cat_n,
                }

        summary = RetrievalEvaluationSummary(
            total_cases=len(dataset),
            answerable_cases=n_ans,
            unanswerable_cases=n_unans,
            recall_at_1=mean_r1,
            recall_at_3=mean_r3,
            recall_at_5=mean_r5,
            precision_at_1=mean_p1,
            precision_at_3=mean_p3,
            precision_at_5=mean_p5,
            mrr=mrr,
            unanswerable_empty_retrieval_count=empty_unans_count,
            unanswerable_non_empty_retrieval_count=non_empty_unans_count,
            unanswerable_empty_retrieval_rate=empty_unans_rate,
            category_metrics=category_metrics,
            per_question_results=per_question_results,
        )

        return summary
