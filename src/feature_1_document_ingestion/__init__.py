# Feature 1: Document Ingestion + Chunking.
# Re-export the public pieces so callers (e.g. main.py) can do:
#   from src.feature_1_document_ingestion import run_ingestion, RecursiveChunker

from .pipeline import run_ingestion, IngestionResult
from .chunking import (
    FixedSizeChunker,
    SentenceChunker,
    SlidingWindowChunker,
    RecursiveChunker,
    StructureAwareChunker,
    SemanticChunker,
    ParentDocumentChunker,
)
