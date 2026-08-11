from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.ingestion.cli import build_parser, main
from app.ingestion.exceptions import DocumentParsingError
from app.ingestion.service import IngestionService


def test_cli_parser_options():
    """Verify argparse parser requires mutually exclusive --file or --dir."""
    parser = build_parser()

    # Valid --file
    args_file = parser.parse_args(["--file", "doc.pdf"])
    assert args_file.file == "doc.pdf"
    assert args_file.dir is None

    # Valid --dir
    args_dir = parser.parse_args(["--dir", "./data"])
    assert args_dir.dir == "./data"
    assert args_dir.file is None


def test_cli_file_and_dir_mutually_exclusive():
    """Test 3: Verify --file and --dir cannot be supplied together."""
    result = main(["--file", "doc.pdf", "--dir", "./data"])
    assert result != 0


def test_cli_neither_file_nor_dir_rejected():
    """Test 4: Verify providing neither --file nor --dir is rejected."""
    result = main([])
    assert result != 0


def test_cli_missing_file_rejected(tmp_path):
    """Test 5: Verify missing/non-existent file path is rejected."""
    missing_file = tmp_path / "does_not_exist.pdf"
    mock_service = MagicMock(spec=IngestionService)

    result = main(["--file", str(missing_file)], service=mock_service)

    assert result == 1
    mock_service.ingest_file.assert_not_called()


def test_cli_missing_dir_rejected(tmp_path):
    """Test 5: Verify missing/non-existent directory path is rejected."""
    missing_dir = tmp_path / "missing_dir"
    mock_service = MagicMock(spec=IngestionService)

    result = main(["--dir", str(missing_dir)], service=mock_service)

    assert result == 1
    mock_service.ingest_directory.assert_not_called()


def test_cli_file_successful(tmp_path, capsys):
    """Test 1, 6, 7: Verify --file works, uses IngestionService, and prints statistics."""
    dummy_file = tmp_path / "test.md"
    dummy_file.write_text("# Hello World")

    mock_service = MagicMock(spec=IngestionService)
    mock_service.ingest_file.return_value = {
        "documents_processed": 1,
        "chunks_created": 2,
        "chunks_upserted": 2,
    }

    result = main(["--file", str(dummy_file)], service=mock_service)

    assert result == 0
    mock_service.ingest_file.assert_called_once_with(dummy_file.resolve())

    captured = capsys.readouterr()
    assert "INGESTION SUCCESSFUL" in captured.out
    assert "Documents Processed: 1" in captured.out
    assert "Chunks Created:     2" in captured.out
    assert "Chunks Upserted:    2" in captured.out


def test_cli_dir_successful(tmp_path, capsys):
    """Test 2, 6, 7: Verify --dir works, uses IngestionService, and prints statistics."""
    dummy_dir = tmp_path / "docs"
    dummy_dir.mkdir()

    mock_service = MagicMock(spec=IngestionService)
    mock_service.ingest_directory.return_value = {
        "documents_processed": 3,
        "chunks_created": 5,
        "chunks_upserted": 5,
    }

    result = main(["--dir", str(dummy_dir)], service=mock_service)

    assert result == 0
    mock_service.ingest_directory.assert_called_once_with(dummy_dir.resolve())

    captured = capsys.readouterr()
    assert "INGESTION SUCCESSFUL" in captured.out
    assert "Documents Processed: 3" in captured.out
    assert "Chunks Created:     5" in captured.out
    assert "Chunks Upserted:    5" in captured.out


def test_cli_failure_exits_nonzero(tmp_path, capsys):
    """Test 8: Verify ingestion failure exits non-zero without unhandled crash."""
    dummy_file = tmp_path / "corrupt.pdf"
    dummy_file.write_text("corrupt content")

    mock_service = MagicMock(spec=IngestionService)
    mock_service.ingest_file.side_effect = DocumentParsingError("Corrupt PDF file")

    result = main(["--file", str(dummy_file)], service=mock_service)

    assert result == 1
    captured = capsys.readouterr()
    assert "Ingestion Failure Error: Corrupt PDF file" in captured.err
