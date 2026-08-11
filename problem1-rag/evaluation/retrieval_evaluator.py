import logging
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.retrieval.base import BaseRetriever
from app.retrieval.service import VectorRetriever
from evaluation.loader import load_evaluation_dataset
from evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
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
        all_latencies_ms: List[float] = []

        for case in dataset:
            q_id = case["id"]
            question = case["question"]
            category = case["category"]
            rel_docs = case.get("relevant_documents", [])
            rel_chunks = case.get("relevant_chunk_ids", [])

            # Invoke retrieval service directly with timing measurement
            start_time = time.perf_counter()
            retrieved_chunks = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                relevance_threshold=relevance_threshold,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            all_latencies_ms.append(elapsed_ms)

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
                    hit_rate_at_1=0.0,
                    hit_rate_at_3=0.0,
                    hit_rate_at_5=0.0,
                    ndcg_at_1=0.0,
                    ndcg_at_3=0.0,
                    ndcg_at_5=0.0,
                    reciprocal_rank=0.0,
                    retrieval_latency_ms=round(elapsed_ms, 2),
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

                hr1 = hit_rate_at_k(retrieved_chunk_ids, rel_chunks, 1)
                hr3 = hit_rate_at_k(retrieved_chunk_ids, rel_chunks, 3)
                hr5 = hit_rate_at_k(retrieved_chunk_ids, rel_chunks, 5)

                ndcg1 = ndcg_at_k(retrieved_chunk_ids, rel_chunks, 1)
                ndcg3 = ndcg_at_k(retrieved_chunk_ids, rel_chunks, 3)
                ndcg5 = ndcg_at_k(retrieved_chunk_ids, rel_chunks, 5)

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
                    hit_rate_at_1=hr1,
                    hit_rate_at_3=hr3,
                    hit_rate_at_5=hr5,
                    ndcg_at_1=ndcg1,
                    ndcg_at_3=ndcg3,
                    ndcg_at_5=ndcg5,
                    reciprocal_rank=rr,
                    retrieval_latency_ms=round(elapsed_ms, 2),
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

            mean_hr1 = sum(c.hit_rate_at_1 for c in answerable_cases) / n_ans
            mean_hr3 = sum(c.hit_rate_at_3 for c in answerable_cases) / n_ans
            mean_hr5 = sum(c.hit_rate_at_5 for c in answerable_cases) / n_ans

            mean_ndcg1 = sum(c.ndcg_at_1 for c in answerable_cases) / n_ans
            mean_ndcg3 = sum(c.ndcg_at_3 for c in answerable_cases) / n_ans
            mean_ndcg5 = sum(c.ndcg_at_5 for c in answerable_cases) / n_ans

            mrr = mean_reciprocal_rank([c.reciprocal_rank for c in answerable_cases])
        else:
            mean_r1 = mean_r3 = mean_r5 = 0.0
            mean_p1 = mean_p3 = mean_p5 = 0.0
            mean_hr1 = mean_hr3 = mean_hr5 = 0.0
            mean_ndcg1 = mean_ndcg3 = mean_ndcg5 = 0.0
            mrr = 0.0

        # Latency percentiles (p50 and p95)
        if all_latencies_ms:
            sorted_lat = sorted(all_latencies_ms)
            p50_idx = int(0.50 * (len(sorted_lat) - 1))
            p95_idx = int(0.95 * (len(sorted_lat) - 1))
            p50_lat = round(sorted_lat[p50_idx], 2)
            p95_lat = round(sorted_lat[p95_idx], 2)
        else:
            p50_lat = p95_lat = 0.0

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
                    "hit_rate_at_1": sum(i.hit_rate_at_1 for i in items) / cat_n,
                    "hit_rate_at_3": sum(i.hit_rate_at_3 for i in items) / cat_n,
                    "hit_rate_at_5": sum(i.hit_rate_at_5 for i in items) / cat_n,
                    "ndcg_at_1": sum(i.ndcg_at_1 for i in items) / cat_n,
                    "ndcg_at_3": sum(i.ndcg_at_3 for i in items) / cat_n,
                    "ndcg_at_5": sum(i.ndcg_at_5 for i in items) / cat_n,
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
            hit_rate_at_1=mean_hr1,
            hit_rate_at_3=mean_hr3,
            hit_rate_at_5=mean_hr5,
            ndcg_at_1=mean_ndcg1,
            ndcg_at_3=mean_ndcg3,
            ndcg_at_5=mean_ndcg5,
            mrr=mrr,
            p50_retrieval_latency_ms=p50_lat,
            p95_retrieval_latency_ms=p95_lat,
            unanswerable_empty_retrieval_count=empty_unans_count,
            unanswerable_non_empty_retrieval_count=non_empty_unans_count,
            unanswerable_empty_retrieval_rate=empty_unans_rate,
            category_metrics=category_metrics,
            per_question_results=per_question_results,
        )

        return summary
