# Shared paths and defaults for all feature benchmarks.

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOC = os.path.join(ROOT, "docs", "sample.txt")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
