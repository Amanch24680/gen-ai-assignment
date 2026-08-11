import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure problem1-rag root is in sys.path when executed directly or via -m
package_dir = Path(__file__).resolve().parent.parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from evaluation.retrieval_evaluator import RetrievalEvaluator

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Command-line Retrieval Evaluation Runner for Cost-Efficient RAG Application."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K retrieval parameter to evaluate (default: 5).",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=None,
        help="Optional relevance threshold override (0.0 to 1.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save full evaluation results JSON.",
    )
    return parser


def main(args: Optional[List[str]] = None, evaluator: Optional[RetrievalEvaluator] = None) -> int:
    """
    CLI entrypoint function for retrieval evaluation.
    Parses CLI arguments, runs RetrievalEvaluator, prints summary report,
    and saves detailed results to output file if requested.
    """
    parser = build_parser()

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    if parsed_args.k <= 0:
        print(f"Error: --k must be a positive integer (> 0), got {parsed_args.k}", file=sys.stderr)
        return 1

    if parsed_args.relevance_threshold is not None and not (0.0 <= parsed_args.relevance_threshold <= 1.0):
        print(
            f"Error: --relevance-threshold must be between 0.0 and 1.0, got {parsed_args.relevance_threshold}",
            file=sys.stderr,
        )
        return 1

    eval_runner = evaluator or RetrievalEvaluator()

    try:
        summary = eval_runner.evaluate(
            top_k=parsed_args.k,
            relevance_threshold=parsed_args.relevance_threshold,
        )

        print("-" * 50)
        print("RAG RETRIEVAL EVALUATION")
        print("-" * 50)
        print(f"Dataset cases: {summary.total_cases}")
        print(f"Answerable cases: {summary.answerable_cases}")
        print(f"Unanswerable cases: {summary.unanswerable_cases}\n")

        print(f"Recall@1: {summary.recall_at_1:.4f}")
        print(f"Recall@3: {summary.recall_at_3:.4f}")
        print(f"Recall@5: {summary.recall_at_5:.4f}\n")

        print(f"Precision@1: {summary.precision_at_1:.4f}")
        print(f"Precision@3: {summary.precision_at_3:.4f}")
        print(f"Precision@5: {summary.precision_at_5:.4f}\n")

        print(f"MRR: {summary.mrr:.4f}\n")

        print(f"Unanswerable empty retrieval rate: {summary.unanswerable_empty_retrieval_rate:.4f}")
        print("-" * 50)

        if parsed_args.output:
            out_path = Path(parsed_args.output).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary.model_dump(), f, indent=2)
            print(f"Detailed evaluation output saved to: {out_path}")

        return 0

    except Exception as exc:
        logger.error(f"Retrieval evaluation failed: {exc}")
        print(f"Evaluation Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
