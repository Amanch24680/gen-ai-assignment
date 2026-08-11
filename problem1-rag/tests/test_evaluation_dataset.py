from pathlib import Path
import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.parsers import DocumentLoader
from evaluation import load_evaluation_dataset, DATASET_PATH


def test_dataset_file_exists():
    """Verify dataset.json exists and is valid JSON."""
    assert DATASET_PATH.exists()
    dataset = load_evaluation_dataset()
    assert isinstance(dataset, list)


def test_dataset_case_count():
    """Verify dataset contains 15-30 cases (target ~20)."""
    dataset = load_evaluation_dataset()
    assert 15 <= len(dataset) <= 30
    assert len(dataset) == 20


def test_dataset_unique_ids():
    """Verify all question IDs and question texts are unique."""
    dataset = load_evaluation_dataset()
    ids = [item["id"] for item in dataset]
    assert len(ids) == len(set(ids))

    questions = [item["question"].strip().lower() for item in dataset]
    assert len(questions) == len(set(questions))


def test_dataset_required_fields_and_types():
    """Verify required fields exist on every case."""
    dataset = load_evaluation_dataset()
    required_keys = {"id", "question", "category", "ground_truth_answer", "relevant_documents", "relevant_chunk_ids"}

    for case in dataset:
        assert set(case.keys()) == required_keys
        assert isinstance(case["id"], str)
        assert isinstance(case["question"], str) and len(case["question"]) > 5
        assert isinstance(case["category"], str)
        assert isinstance(case["relevant_documents"], list)
        assert isinstance(case["relevant_chunk_ids"], list)

        if case["category"] == "unanswerable":
            assert case["ground_truth_answer"] is None
            assert len(case["relevant_documents"]) == 0
            assert len(case["relevant_chunk_ids"]) == 0
        else:
            assert isinstance(case["ground_truth_answer"], str) and len(case["ground_truth_answer"]) > 0
            assert len(case["relevant_documents"]) > 0
            assert len(case["relevant_chunk_ids"]) > 0


def test_dataset_multi_chunk_cases_require_multiple_chunks():
    """Verify that multi_chunk category cases specify at least 2 relevant chunk IDs."""
    dataset = load_evaluation_dataset()
    multi_chunk_cases = [c for c in dataset if c["category"] == "multi_chunk"]
    assert len(multi_chunk_cases) >= 1
    for case in multi_chunk_cases:
        assert len(case["relevant_chunk_ids"]) >= 2, f"multi_chunk case {case['id']} has fewer than 2 chunk IDs"


def test_dataset_chunk_references_exist_in_corpus():
    """Verify that all referenced document IDs and chunk IDs in dataset.json exist in actual corpus."""
    dataset = load_evaluation_dataset()

    loader = DocumentLoader()
    chunker = TextChunker()
    base_dir = Path(__file__).resolve().parent.parent.parent

    files_to_ingest = [
        base_dir / "Gen AI_assignment.pdf",
        base_dir / "problem1-rag" / "tests" / "fixtures" / "sample.pdf",
        base_dir / "problem1-rag" / "tests" / "fixtures" / "sample.html",
        base_dir / "problem1-rag" / "tests" / "fixtures" / "sample.md",
    ]

    actual_doc_ids = set()
    actual_chunk_ids = set()

    for fp in files_to_ingest:
        if not fp.exists():
            continue
        docs = loader.load_document(fp)
        for doc in docs:
            actual_doc_ids.add(doc.doc_id)
            chunks = chunker.chunk_document(doc)
            for c in chunks:
                actual_chunk_ids.add(c.chunk_id)

    for case in dataset:
        for doc_id in case["relevant_documents"]:
            assert doc_id in actual_doc_ids, f"Referenced doc_id {doc_id} in {case['id']} does not exist in corpus"
        for chunk_id in case["relevant_chunk_ids"]:
            assert chunk_id in actual_chunk_ids, f"Referenced chunk_id {chunk_id} in {case['id']} does not exist in corpus"
