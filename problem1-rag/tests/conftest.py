import sys
from pathlib import Path

# Ensure problem1-rag directory is in sys.path for test imports
problem1_rag_dir = Path(__file__).resolve().parent.parent
if str(problem1_rag_dir) not in sys.path:
    sys.path.insert(0, str(problem1_rag_dir))
