# Shared chunker registry — used by Feature 1 and Feature 2 benchmarks.

from __future__ import annotations

from typing import Callable

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from ..feature_1_document_ingestion import (
    FixedSizeChunker,
    ParentDocumentChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    SlidingWindowChunker,
    StructureAwareChunker,
)
from ..feature_1_document_ingestion.chunking.base_chunker import BaseChunker

ChunkerFactory = Callable[[], BaseChunker]

CHUNKERS: dict[str, ChunkerFactory] = {
    "fixed": lambda: FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=0),
    "sentence": lambda: SentenceChunker(max_chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
    "sliding": lambda: SlidingWindowChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
    "recursive": lambda: RecursiveChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
    "structure": lambda: StructureAwareChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
    "semantic": lambda: SemanticChunker(max_chunk_size=CHUNK_SIZE, similarity_threshold=0.15),
    "parent": lambda: ParentDocumentChunker(
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=250,
        child_chunk_overlap=40,
    ),
}

CHUNKER_KEYS: tuple[str, ...] = tuple(CHUNKERS.keys())


def get_chunker(name: str) -> BaseChunker:
    if name not in CHUNKERS:
        known = ", ".join(CHUNKER_KEYS)
        raise ValueError(f"Unknown chunker '{name}'. Choose one of: {known}")
    return CHUNKERS[name]()
