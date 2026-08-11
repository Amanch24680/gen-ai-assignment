import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure package directory is in sys.path
package_dir = Path(__file__).resolve().parent.parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from app.bias import run_complete_bias_suite
from app.config import settings
from app.evaluator import Evaluator
from app.judge import JudgeClient
from app.schemas import ABComparisonItem, EvaluationItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for LLM-as-Judge pipeline."""
    parser = argparse.ArgumentParser(
        description="Command-line Runner for Problem 2 LLM-as-Judge Evaluation Pipeline."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Path to JSON test suite file (e.g. datasets/suite.json).",
    )
    parser.add_argument(
        "--ab",
        type=str,
        default=None,
        help="Path to JSON A/B comparison test suite file (e.g. datasets/ab_suite.json).",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Run comprehensive bias measurement probe suite.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run suite evaluation, A/B comparison, and bias probe suite sequentially.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON evaluation report.",
    )
    return parser


def load_items_from_json(path: Path) -> List[EvaluationItem]:
    """Load EvaluationItem list from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Suite dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvaluationItem(**d) for d in data]


def load_ab_items_from_json(path: Path) -> List[ABComparisonItem]:
    """Load ABComparisonItem list from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"A/B dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ABComparisonItem(**d) for d in data]


def run_suite_command(suite_path: Path, output_path: Optional[Path], evaluator: Evaluator) -> int:
    """Run single-item evaluation suite."""
    print("-" * 60)
    print("RUNNING LLM-AS-JUDGE SUITE EVALUATION")
    print("-" * 60)
    print(f"Judge Model: {evaluator.judge_client.judge_model}")
    print(f"Suite File: {suite_path}\n")

    items = load_items_from_json(suite_path)
    report = evaluator.evaluate_suite(items)

    print(f"Total Cases: {report.total_cases}")
    print(f"Passed: {report.passed_cases}")
    print(f"Failed: {report.failed_cases}")
    print(f"Pass Rate: {report.pass_rate * 100:.1f}%")
    print(f"Mean Overall Score: {report.mean_overall_score:.2f} / 5.0")
    print("\nMean Criterion Scores:")
    for crit, score in report.mean_criterion_scores.items():
        print(f"  - {crit}: {score:.2f}")
    print("-" * 60)

    out_file = output_path or (settings.results_dir / "suite_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
    print(f"Suite evaluation report saved to: {out_file}\n")
    return 0


def run_ab_command(ab_path: Path, output_path: Optional[Path], evaluator: Evaluator) -> int:
    """Run pairwise A/B comparison evaluation suite."""
    print("-" * 60)
    print("RUNNING PAIRWISE A/B COMPARISON EVALUATION")
    print("-" * 60)
    print(f"Judge Model: {evaluator.judge_client.judge_model}")
    print(f"A/B Suite File: {ab_path}\n")

    items = load_ab_items_from_json(ab_path)
    report = evaluator.evaluate_ab_suite(items, swap_order=False)

    print(f"Total Cases: {report.total_cases}")
    print(f"Candidate A Wins: {report.wins_a}")
    print(f"Candidate B Wins: {report.wins_b}")
    print(f"Ties: {report.ties}")
    print(f"Overall Winner: {report.overall_winner}")
    print("-" * 60)

    out_file = output_path or (settings.results_dir / "ab_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
    print(f"A/B comparison report saved to: {out_file}\n")
    return 0


def run_bias_command(output_path: Optional[Path], evaluator: Evaluator) -> int:
    """Run bias measurement probes."""
    print("-" * 60)
    print("RUNNING LLM-AS-JUDGE BIAS MEASUREMENT PROBES")
    print("-" * 60)

    suite_path = settings.datasets_dir / "suite.json"
    ab_path = settings.datasets_dir / "ab_suite.json"

    items = load_items_from_json(suite_path)
    ab_items = load_ab_items_from_json(ab_path)

    normal_item = next((i for i in items if i.category == "normal_correct"), items[0])
    padded_item = next((i for i in items if i.category == "verbose_correct"), items[0])
    confident_wrong_items = [i for i in items if i.category == "confidently_wrong"]

    # Run single suite to gather score distribution
    suite_report = evaluator.evaluate_suite(items)
    all_scores = [v.overall_score for v in suite_report.verdicts]

    bias_report = run_complete_bias_suite(
        evaluator=evaluator,
        ab_items=ab_items,
        normal_item=normal_item,
        padded_item=padded_item,
        confident_wrong_items=confident_wrong_items,
        all_suite_scores=all_scores,
    )

    print(f"1. Position Bias Flip Rate: {bias_report.position_bias.flip_rate * 100:.1f}% ({bias_report.position_bias.flips}/{bias_report.position_bias.total_pairs} flips)")
    print(f"2. Verbosity Score Delta: {bias_report.verbosity_bias.score_delta:+.2f} (Normal: {bias_report.verbosity_bias.normal_mean_score:.2f}, Verbose: {bias_report.verbosity_bias.verbose_mean_score:.2f})")
    print(f"3. Sycophancy Detection: {bias_report.sycophancy_bias.detected_correctly}/{bias_report.sycophancy_bias.total_cases} detected (Sycophancy Rate: {bias_report.sycophancy_bias.sycophancy_rate * 100:.1f}%)")
    print(f"4. Score Spread (Std Dev): {bias_report.score_clustering.score_std_dev:.4f} (Min: {bias_report.score_clustering.min_score:.1f}, Max: {bias_report.score_clustering.max_score:.1f})")
    print("-" * 60)

    out_file = output_path or (settings.results_dir / "bias_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bias_report.model_dump(), f, indent=2)
    print(f"Bias measurement report saved to: {out_file}\n")
    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint for Problem 2 LLM-as-Judge pipeline."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    evaluator = Evaluator()

    if parsed.all:
        suite_p = Path(parsed.suite) if parsed.suite else (settings.datasets_dir / "suite.json")
        ab_p = Path(parsed.ab) if parsed.ab else (settings.datasets_dir / "ab_suite.json")
        run_suite_command(suite_p, settings.results_dir / "suite_results.json", evaluator)
        run_ab_command(ab_p, settings.results_dir / "ab_results.json", evaluator)
        run_bias_command(settings.results_dir / "bias_results.json", evaluator)
        return 0

    if parsed.suite:
        out_p = Path(parsed.output) if parsed.output else None
        return run_suite_command(Path(parsed.suite), out_p, evaluator)

    if parsed.ab:
        out_p = Path(parsed.output) if parsed.output else None
        return run_ab_command(Path(parsed.ab), out_p, evaluator)

    if parsed.bias:
        out_p = Path(parsed.output) if parsed.output else None
        return run_bias_command(out_p, evaluator)

    # Default if no arguments provided: run all
    suite_p = settings.datasets_dir / "suite.json"
    ab_p = settings.datasets_dir / "ab_suite.json"
    run_suite_command(suite_p, settings.results_dir / "suite_results.json", evaluator)
    run_ab_command(ab_p, settings.results_dir / "ab_results.json", evaluator)
    run_bias_command(settings.results_dir / "bias_results.json", evaluator)
    return 0


if __name__ == "__main__":
    sys.exit(main())
