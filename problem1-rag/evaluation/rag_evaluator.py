import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.api.dependencies import get_rag_service
from app.rag.service import RAGService
from evaluation.loader import load_evaluation_dataset
from evaluation.metrics import (
    answer_token_f1,
    citation_coverage_score,
    context_support_score,
    unanswerable_safe_handling,
)
from evaluation.schemas import RAGAnswerEvaluationSummary, RAGAnswerQuestionResult

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    End-to-End RAG Answer Pipeline Evaluator.
    Evaluates complete RAG pipeline (Retrieval + LLM Generation) against ground truth dataset.
    Uses the production RAGService interface.
    """

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        dataset_path: Optional[Path] = None,
    ):
        self.rag_service = rag_service or get_rag_service()
        self.dataset_path = dataset_path

    def evaluate(
        self,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        limit: Optional[int] = None,
        case_ids: Optional[List[str]] = None,
    ) -> RAGAnswerEvaluationSummary:
        """
        Run end-to-end RAG evaluation across dataset questions.
        Calculates answer F1, context support, citation coverage, and latency metrics.
        """
        dataset = load_evaluation_dataset(self.dataset_path)
        
        if case_ids:
            target_ids = set(case_ids)
            dataset = [c for c in dataset if c["id"] in target_ids]

        if limit is not None and limit > 0:
            dataset = dataset[:limit]

        logger.info(f"Loaded {len(dataset)} evaluation cases for RAG answer evaluation.")

        per_question_results: List[RAGAnswerQuestionResult] = []
        answerable_cases: List[RAGAnswerQuestionResult] = []
        unanswerable_cases: List[RAGAnswerQuestionResult] = []

        successful_generations = 0
        failed_generations = 0

        for case in dataset:
            q_id = case["id"]
            question = case["question"]
            category = case["category"]
            gt_answer = case.get("ground_truth_answer")
            rel_chunks = case.get("relevant_chunk_ids", [])

            try:
                # Invoke production RAG pipeline (retrieval + generation + metrics)
                rag_response = self.rag_service.query(
                    query=question,
                    top_k=top_k,
                    relevance_threshold=relevance_threshold,
                )

                generated_answer = rag_response.answer or ""
                citations = rag_response.citations or []
                cited_chunk_ids = [c.chunk_id for c in citations]
                has_relevant_context = rag_response.has_relevant_context
                latency_ms = rag_response.latency_ms

                # Context texts from citations for context support calculation
                retrieved_texts = [c.snippet for c in citations if c.snippet]

                successful_generations += 1
                success = True
                error_msg = None

            except Exception as exc:
                logger.error(f"RAG query failed for [{q_id}]: {exc}")
                failed_generations += 1
                generated_answer = ""
                citations = []
                cited_chunk_ids = []
                has_relevant_context = False
                latency_ms = 0.0
                retrieved_texts = []
                success = False
                error_msg = str(exc)

            if category == "unanswerable":
                is_safe = unanswerable_safe_handling(generated_answer, has_relevant_context)
                result = RAGAnswerQuestionResult(
                    question_id=q_id,
                    question=question,
                    category=category,
                    ground_truth_answer=gt_answer,
                    generated_answer=generated_answer,
                    retrieved_chunk_ids=cited_chunk_ids,
                    cited_chunk_ids=cited_chunk_ids,
                    has_relevant_context=has_relevant_context,
                    latency_ms=latency_ms,
                    answer_f1=0.0,
                    context_support=0.0,
                    citation_coverage=0.0,
                    is_unanswerable=True,
                    unanswerable_safe_handled=is_safe,
                    success=success,
                    error_message=error_msg,
                )
                unanswerable_cases.append(result)
            else:
                f1 = answer_token_f1(generated_answer, gt_answer)
                c_supp = context_support_score(generated_answer, retrieved_texts)
                c_cov = citation_coverage_score(citations, cited_chunk_ids or rel_chunks)

                result = RAGAnswerQuestionResult(
                    question_id=q_id,
                    question=question,
                    category=category,
                    ground_truth_answer=gt_answer,
                    generated_answer=generated_answer,
                    retrieved_chunk_ids=cited_chunk_ids,
                    cited_chunk_ids=cited_chunk_ids,
                    has_relevant_context=has_relevant_context,
                    latency_ms=latency_ms,
                    answer_f1=f1,
                    context_support=c_supp,
                    citation_coverage=c_cov,
                    is_unanswerable=False,
                    unanswerable_safe_handled=False,
                    success=success,
                    error_message=error_msg,
                )
                answerable_cases.append(result)

            per_question_results.append(result)

        # Aggregate answerable metrics
        n_ans = len(answerable_cases)
        if n_ans > 0:
            mean_f1 = sum(c.answer_f1 for c in answerable_cases) / n_ans
            mean_supp = sum(c.context_support for c in answerable_cases) / n_ans
            mean_cov = sum(c.citation_coverage for c in answerable_cases) / n_ans
        else:
            mean_f1 = mean_supp = mean_cov = 0.0

        # Unanswerable metrics
        n_unans = len(unanswerable_cases)
        safe_unans_count = sum(1 for c in unanswerable_cases if c.unanswerable_safe_handled)
        safe_unans_rate = (safe_unans_count / n_unans) if n_unans > 0 else 0.0

        # Latency metrics across all cases
        latencies = [c.latency_ms for c in per_question_results if c.latency_ms > 0]
        if latencies:
            avg_lat = round(sum(latencies) / len(latencies), 2)
            med_lat = round(statistics.median(latencies), 2)
            min_lat = round(min(latencies), 2)
            max_lat = round(max(latencies), 2)
        else:
            avg_lat = med_lat = min_lat = max_lat = 0.0

        # Category breakdowns
        category_groups: Dict[str, List[RAGAnswerQuestionResult]] = {}
        for res in per_question_results:
            category_groups.setdefault(res.category, []).append(res)

        category_metrics: Dict[str, Dict[str, float]] = {}
        for cat, items in category_groups.items():
            if cat == "unanswerable":
                safe_cnt = sum(1 for item in items if item.unanswerable_safe_handled)
                category_metrics[cat] = {
                    "count": float(len(items)),
                    "safe_handling_rate": (safe_cnt / len(items)) if items else 0.0,
                }
            else:
                cat_n = len(items)
                category_metrics[cat] = {
                    "count": float(cat_n),
                    "answer_f1": sum(i.answer_f1 for i in items) / cat_n,
                    "context_support": sum(i.context_support for i in items) / cat_n,
                    "citation_coverage": sum(i.citation_coverage for i in items) / cat_n,
                }

        summary = RAGAnswerEvaluationSummary(
            total_cases=len(dataset),
            answerable_cases=n_ans,
            unanswerable_cases=n_unans,
            successful_generations=successful_generations,
            failed_generations=failed_generations,
            mean_answer_f1=round(mean_f1, 4),
            mean_context_support=round(mean_supp, 4),
            mean_citation_coverage=round(mean_cov, 4),
            unanswerable_safe_handling_rate=round(safe_unans_rate, 4),
            average_latency_ms=avg_lat,
            median_latency_ms=med_lat,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            category_metrics=category_metrics,
            per_question_results=per_question_results,
        )

        return summary
