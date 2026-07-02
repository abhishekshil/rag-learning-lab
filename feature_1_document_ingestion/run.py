import os
import sys

# Make sure Python can find our src/ package regardless of where we call from
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import run_ingestion
from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    SlidingWindowChunker,
    RecursiveChunker,
    StructureAwareChunker,
    SemanticChunker,
    ParentDocumentChunker,
)


SAMPLE_DOC = os.path.join(os.path.dirname(__file__), "sample_docs", "sample.txt")

# ── Configuration ─────────────────────────────────────────────────────────────
# Most strategies use the same target size so the comparison is fair.
CHUNK_SIZE = 500
OVERLAP = 50

CHUNKERS = [
    FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=0),
    SentenceChunker(max_chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP),
    SlidingWindowChunker(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP),
    RecursiveChunker(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP),
    StructureAwareChunker(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP),
    SemanticChunker(max_chunk_size=CHUNK_SIZE, similarity_threshold=0.15),
    ParentDocumentChunker(
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=250,
        child_chunk_overlap=40,
    ),
]
# ──────────────────────────────────────────────────────────────────────────────


def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_comparison_table(results: list) -> None:
    print("\n")
    print_separator("═")
    print("  CHUNKING STRATEGY COMPARISON")
    print_separator("═")
    print(f"  {'Strategy':<45} {'Chunks':>6} {'Avg':>6} {'Min':>6} {'Max':>6}")
    print_separator()
    for r in results:
        print(
            f"  {r.strategy_name:<45} "
            f"{r.chunk_count:>6} "
            f"{r.avg_chunk_size:>6.0f} "
            f"{r.min_chunk_size:>6} "
            f"{r.max_chunk_size:>6}"
        )
    print_separator("═")
    print("  Avg / Min / Max = characters per chunk")
    print_separator("═")
    print()


def print_chunks(result) -> None:
    print_separator("─")
    print(f"  Strategy: {result.strategy_name}")
    print(f"  File: {os.path.basename(result.file_path)}")
    print(f"  Raw chars: {result.raw_char_count}  →  Clean chars: {result.clean_char_count}")
    print(f"  Total chunks: {result.chunk_count}")
    print_separator("─")

    for chunk in result.chunks:
        print(f"\n  [Chunk {chunk.index}]  chars {chunk.char_start}–{chunk.char_end}")
        # Indent each line of the chunk text for readability
        for line in chunk.text.split("\n"):
            print(f"    {line}")

    print()


def main():
    print("\n  Feature 1: Document Ingestion Pipeline")
    print("  Running all chunking strategies on:", SAMPLE_DOC)

    results = []

    for chunker in CHUNKERS:
        print(f"\n  Running {chunker.strategy_name()} ...", end=" ", flush=True)
        result = run_ingestion(SAMPLE_DOC, chunker)
        results.append(result)
        print(f"done — {result.chunk_count} chunks")

    # Summary table first so you see the big picture before the detail
    print_comparison_table(results)

    # Then show every chunk from every strategy
    for result in results:
        print_chunks(result)


if __name__ == "__main__":
    main()
