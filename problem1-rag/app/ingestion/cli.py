import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure problem1-rag directory is in sys.path for direct module execution
package_dir = Path(__file__).resolve().parent.parent.parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from app.ingestion.service import IngestionService

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser with mutually exclusive --file and --dir options."""
    parser = argparse.ArgumentParser(
        description="Command-line document ingestion tool for Cost-Efficient RAG Pipeline."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type=str,
        help="Path to a single document file (PDF, HTML, MD) to ingest.",
    )
    group.add_argument(
        "--dir",
        type=str,
        help="Path to a directory containing document files to ingest.",
    )
    return parser


def main(args: Optional[List[str]] = None, service: Optional[IngestionService] = None) -> int:
    """
    CLI entrypoint function.
    Parses arguments, instantiates IngestionService, runs ingestion, and prints statistics.
    Returns 0 on success, 1 on failure.
    """
    parser = build_parser()

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as exc:
        # argparse raises SystemExit on invalid options or --help
        return exc.code if isinstance(exc.code, int) else 1

    ingestion_service = service or IngestionService()

    try:
        if parsed_args.file:
            file_path = Path(parsed_args.file).resolve()
            if not file_path.exists():
                print(f"Error: File does not exist: {file_path}", file=sys.stderr)
                return 1
            if not file_path.is_file():
                print(f"Error: Path is not a file: {file_path}", file=sys.stderr)
                return 1

            stats = ingestion_service.ingest_file(file_path)

        elif parsed_args.dir:
            dir_path = Path(parsed_args.dir).resolve()
            if not dir_path.exists():
                print(f"Error: Directory does not exist: {dir_path}", file=sys.stderr)
                return 1
            if not dir_path.is_dir():
                print(f"Error: Path is not a directory: {dir_path}", file=sys.stderr)
                return 1

            stats = ingestion_service.ingest_directory(dir_path)

        else:
            print("Error: Either --file or --dir must be specified.", file=sys.stderr)
            return 1

        print("=" * 50)
        print("INGESTION SUCCESSFUL")
        print("=" * 50)
        print(f"Documents Processed: {stats['documents_processed']}")
        print(f"Chunks Created:     {stats['chunks_created']}")
        print(f"Chunks Upserted:    {stats['chunks_upserted']}")
        print("=" * 50)
        return 0

    except Exception as exc:
        logger.error(f"Ingestion process failed: {exc}")
        print(f"Ingestion Failure Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
