# pipeline.py
# Responsibility: Orchestrate Feature 2 end to end.
#
#   Feature 1 (reused)                   Feature 2 (this feature)
#   ┌───────────────────────────┐        ┌──────────────────────────────┐
#   │ load → clean → chunk      │  --->  │ embed → store → top-k search │
#   └───────────────────────────┘        └──────────────────────────────┘
#
# Feature 2 is a CONTINUATION of Feature 1: we do NOT re-implement chunking.
# We import Feature 1's ingestion directly (same project, same `src` package),
# take the exact Chunk objects it produces, turn each into a vector with an
# embedder, wrap it as a VectorRecord (vector + text + metadata), and load them
# into a vector store we can query.

import os
import time
from dataclasses import dataclass

from ..feature_1_document_ingestion.pipeline import run_ingestion
from ..feature_1_document_ingestion.chunking.base_chunker import Chunk
from .embedding.base_embedder import BaseEmbedder
from .vector_store.base_store import BaseVectorStore, VectorRecord, SearchResult


@dataclass
class IndexResult:
    """Everything produced by building one searchable index."""
    embedder_name: str
    store: BaseVectorStore
    chunk_count: int
    dimension: int
    embed_seconds: float


def chunks_to_records(
    chunks: list[Chunk],
    embedder: BaseEmbedder,
    source_file: str,
) -> tuple[list[VectorRecord], float]:
    """
    Embed every chunk and pair each vector with its text + metadata.

    Returns:
        (records, embed_seconds)
    """
    texts = [c.text for c in chunks]

    start = time.perf_counter()
    vectors = embedder.embed(texts)
    embed_seconds = time.perf_counter() - start

    source_name = os.path.basename(source_file)
    records: list[VectorRecord] = []
    for chunk, vector in zip(chunks, vectors):
        records.append(
            VectorRecord(
                id=f"chunk-{chunk.index}",
                vector=vector,
                text=chunk.text,
                metadata={
                    "source": source_name,
                    "chunk_index": chunk.index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
            )
        )
    return records, embed_seconds


def build_index(
    file_path: str,
    chunker,
    embedder: BaseEmbedder,
    store: BaseVectorStore,
) -> IndexResult:
    """
    Full Feature 2 pipeline for one embedder + store combination.

    Step 1: run Feature 1's ingestion to get chunks.
    Step 2: embed the chunks.
    Step 3: load vectors + metadata into the store.
    """
    ingestion = run_ingestion(file_path, chunker)
    records, embed_seconds = chunks_to_records(ingestion.chunks, embedder, file_path)
    store.add(records)

    return IndexResult(
        embedder_name=embedder.name(),
        store=store,
        chunk_count=len(records),
        dimension=embedder.dimension,
        embed_seconds=embed_seconds,
    )


def search(
    store: BaseVectorStore,
    embedder: BaseEmbedder,
    query: str,
    k: int = 3,
) -> list[SearchResult]:
    """Embed a query with the SAME embedder used to build the index, then search."""
    query_vector = embedder.embed_one(query)
    return store.search(query_vector, k=k)
