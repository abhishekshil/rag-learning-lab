#!/usr/bin/env python3
# main.py
# The single entry point (app) for the whole RAG Learning Lab.
#
# It decides WHICH feature to run and WHICH options to use. The features
# themselves live as libraries under src/feature_* and know nothing about the
# CLI or the sample document — this file wires them together.
#
# Usage:
#   python3 main.py                                  # interactive menu
#   python3 main.py --feature 1 --chunker all
#   python3 main.py --feature 1 --chunker recursive --show-chunks
#   python3 main.py --feature 2 --embedder all --metric cosine
#   python3 main.py --feature 2 --chunker recursive --embedder semantic --no-metrics

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.feature_1_document_ingestion import (
    run_ingestion,
    FixedSizeChunker,
    SentenceChunker,
    SlidingWindowChunker,
    RecursiveChunker,
    StructureAwareChunker,
    SemanticChunker,
    ParentDocumentChunker,
)
from src.feature_2_embeddings_vector_store import (
    build_index,
    chunks_to_records,
    search,
    InMemoryVectorStore,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)

# ── Shared config ─────────────────────────────────────────────────────────────
DOC = os.path.join(ROOT, "docs", "sample.txt")
CHUNK_SIZE = 500
OVERLAP = 50
TOP_K = 3

# Registry of chunking strategies (Feature 1). Each value builds a fresh chunker.
CHUNKERS = {
    "fixed": lambda: FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=0),
    "sentence": lambda: SentenceChunker(max_chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP),
    "sliding": lambda: SlidingWindowChunker(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP),
    "recursive": lambda: RecursiveChunker(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP),
    "structure": lambda: StructureAwareChunker(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP),
    "semantic": lambda: SemanticChunker(max_chunk_size=CHUNK_SIZE, similarity_threshold=0.15),
    "parent": lambda: ParentDocumentChunker(
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=250,
        child_chunk_overlap=40,
    ),
}

# Registry of embedding strategies (Feature 2).
EMBEDDERS = {
    "hashing": lambda: HashingEmbedder(dimension=512),
    "semantic": lambda: SentenceTransformerEmbedder(),
}

METRICS = ("cosine", "dot", "euclidean")

# Queries for Feature 2. Each targets a real section of the doc but uses words
# that mostly DON'T appear verbatim there, to expose keyword vs semantic search.
QUERIES = [
    "How can a model answer questions about recent events it was never trained on?",
    "why repeat a little text between neighboring pieces",
    "how does the system find passages related to a user's question?",
]
# ──────────────────────────────────────────────────────────────────────────────


def sep(char: str = "─", width: int = 78) -> None:
    print(char * width)


def preview(text: str, width: int = 88) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


# ── Feature 1: presentation ───────────────────────────────────────────────────
def print_chunk_table(results: list) -> None:
    print("\n")
    sep("═")
    print("  CHUNKING STRATEGY COMPARISON")
    sep("═")
    print(f"  {'Strategy':<45} {'Chunks':>6} {'Avg':>6} {'Min':>6} {'Max':>6}")
    sep()
    for r in results:
        print(
            f"  {r.strategy_name:<45} "
            f"{r.chunk_count:>6} "
            f"{r.avg_chunk_size:>6.0f} "
            f"{r.min_chunk_size:>6} "
            f"{r.max_chunk_size:>6}"
        )
    sep("═")
    print("  Avg / Min / Max = characters per chunk")
    sep("═")


def print_chunks(result) -> None:
    sep("─")
    print(f"  Strategy: {result.strategy_name}")
    print(f"  Raw chars: {result.raw_char_count}  →  Clean chars: {result.clean_char_count}")
    print(f"  Total chunks: {result.chunk_count}")
    sep("─")
    for chunk in result.chunks:
        print(f"\n  [Chunk {chunk.index}]  chars {chunk.char_start}–{chunk.char_end}")
        for line in chunk.text.split("\n"):
            print(f"    {line}")
    print()


def run_feature_1(chunker_choice: str, show_chunks: bool) -> None:
    names = list(CHUNKERS) if chunker_choice == "all" else [chunker_choice]

    print("\n  Feature 1: Document Ingestion + Chunking")
    print("  Source doc:", DOC)

    results = []
    for name in names:
        chunker = CHUNKERS[name]()
        print(f"\n  Running {chunker.strategy_name()} ...", end=" ", flush=True)
        result = run_ingestion(DOC, chunker)
        results.append(result)
        print(f"done — {result.chunk_count} chunks")

    print_chunk_table(results)

    if show_chunks:
        for result in results:
            print_chunks(result)


# ── Feature 2: presentation ───────────────────────────────────────────────────
def print_embedder_table(index_results: list) -> None:
    print("\n")
    sep("═")
    print("  EMBEDDER COMPARISON")
    sep("═")
    print(f"  {'Embedder':<52} {'Dim':>5} {'Chunks':>7} {'Embed(s)':>9}")
    sep()
    for r in index_results:
        print(
            f"  {r.embedder_name:<52} "
            f"{r.dimension:>5} "
            f"{r.chunk_count:>7} "
            f"{r.embed_seconds:>9.3f}"
        )
    sep("═")
    print("  Dim = vector length   Embed(s) = time to embed all chunks")
    sep("═")


def print_query_results(query: str, labelled_results: list) -> None:
    print()
    sep("─")
    print(f"  QUERY: {query}")
    sep("─")
    for label, results in labelled_results:
        print(f"\n  ▸ {label}")
        if not results:
            print("      (no results)")
            continue
        for rank, res in enumerate(results, start=1):
            meta = res.record.metadata
            print(
                f"      {rank}. score={res.score:+.3f}  "
                f"[{res.record.id} @ chars {meta['char_start']}-{meta['char_end']}]"
            )
            print(f"         {preview(res.record.text)}")
    print()


def run_feature_2(chunker_choice: str, embedder_choice: str, metric: str, show_metrics: bool) -> None:
    print("\n  Feature 2: Embeddings + Vector Store")
    print("  Continuation of Feature 1 — chunks come from Feature 1's ingestion.")
    print(f"  Source doc: {DOC}")
    print(f"  Chunking strategy: {chunker_choice}   Metric: {metric}")

    embedder_names = list(EMBEDDERS) if embedder_choice == "all" else [embedder_choice]

    index_results = []
    stores = []
    for name in embedder_names:
        embedder = EMBEDDERS[name]()
        print(f"\n  Building index with {embedder.name()} ...", end=" ", flush=True)
        store = InMemoryVectorStore(metric=metric)
        result = build_index(DOC, CHUNKERS[chunker_choice](), embedder, store)
        print(f"done — {result.chunk_count} chunks in {result.embed_seconds:.3f}s")
        index_results.append(result)
        stores.append((name, embedder, store))

    print_embedder_table(index_results)

    for query in QUERIES:
        labelled = []
        for name, embedder, store in stores:
            hits = search(store, embedder, query, k=TOP_K)
            tag = "KEYWORD " if name == "hashing" else "SEMANTIC"
            labelled.append((f"{tag} | {embedder.name()}", hits))
        print_query_results(query, labelled)

    if show_metrics and "semantic" in embedder_names:
        _run_metric_comparison(chunker_choice)


def _run_metric_comparison(chunker_choice: str) -> None:
    print("\n")
    sep("═")
    print("  SIMILARITY METRIC COMPARISON (same dense vectors)")
    sep("═")

    embedder = EMBEDDERS["semantic"]()
    ingestion = run_ingestion(DOC, CHUNKERS[chunker_choice]())
    records, _ = chunks_to_records(ingestion.chunks, embedder, DOC)

    query = QUERIES[2]
    labelled = []
    for metric in METRICS:
        store = InMemoryVectorStore(metric=metric)
        store.add(records)
        hits = search(store, embedder, query, k=TOP_K)
        labelled.append((f"metric = {metric}", hits))

    print_query_results(query, labelled)
    print("  Note: with unit-normalized vectors, cosine and dot rank identically;")
    print("  euclidean agrees here because distance and angle are linked on the unit sphere.")
    sep("═")


# ── Interactive menu (used when no CLI flags are given) ────────────────────────
def _ask(prompt: str, default: str, choices=None) -> str:
    hint = f" [{default}]" if not choices else f" ({'/'.join(choices)}) [{default}]"
    try:
        raw = input(f"  {prompt}{hint}: ").strip()
    except EOFError:
        raw = ""
    value = raw or default
    if choices and value not in choices:
        print(f"    '{value}' is not valid, using default '{default}'.")
        return default
    return value


def _ask_bool(prompt: str, default: bool) -> bool:
    d = "y" if default else "n"
    return _ask(prompt, d, ["y", "n"]) == "y"


def interactive_config() -> dict:
    print("\n  RAG Learning Lab")
    sep("─")
    print("  1) Document Ingestion + Chunking")
    print("  2) Embeddings + Vector Store")
    sep("─")

    feature = _ask("Choose feature", "2", ["1", "2"])
    if feature == "1":
        chunker = _ask("Chunking strategy", "all", list(CHUNKERS) + ["all"])
        show_chunks = _ask_bool("Print every chunk", False)
        return {"feature": 1, "chunker": chunker, "show_chunks": show_chunks}

    chunker = _ask("Chunking strategy (source of chunks)", "recursive", list(CHUNKERS))
    embedder = _ask("Embedder", "all", list(EMBEDDERS) + ["all"])
    metric = _ask("Similarity metric", "cosine", list(METRICS))
    show_metrics = _ask_bool("Show metric comparison", True)
    return {
        "feature": 2,
        "chunker": chunker,
        "embedder": embedder,
        "metric": metric,
        "show_metrics": show_metrics,
    }


def parse_args(argv: list) -> dict:
    parser = argparse.ArgumentParser(description="RAG Learning Lab runner")
    parser.add_argument("--feature", type=int, choices=[1, 2], help="Which feature to run")
    parser.add_argument(
        "--chunker",
        choices=list(CHUNKERS) + ["all"],
        help="Chunking strategy. Feature 1 accepts 'all'; Feature 2 needs one.",
    )
    parser.add_argument("--embedder", choices=list(EMBEDDERS) + ["all"], default="all")
    parser.add_argument("--metric", choices=list(METRICS), default="cosine")
    parser.add_argument("--show-chunks", action="store_true", help="Feature 1: print every chunk")
    parser.add_argument("--no-metrics", action="store_true", help="Feature 2: skip metric comparison")
    args = parser.parse_args(argv)

    if args.feature == 1:
        return {
            "feature": 1,
            "chunker": args.chunker or "all",
            "show_chunks": args.show_chunks,
        }
    return {
        "feature": 2,
        "chunker": args.chunker if args.chunker and args.chunker != "all" else "recursive",
        "embedder": args.embedder,
        "metric": args.metric,
        "show_metrics": not args.no_metrics,
    }


def main() -> None:
    argv = sys.argv[1:]
    config = parse_args(argv) if argv else interactive_config()

    if config["feature"] == 1:
        run_feature_1(config["chunker"], config["show_chunks"])
    else:
        run_feature_2(
            config["chunker"],
            config["embedder"],
            config["metric"],
            config["show_metrics"],
        )


if __name__ == "__main__":
    main()
