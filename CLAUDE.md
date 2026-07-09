# CLAUDE.md

Guidance for AI agents working in **RAG Learning Lab** — a build-first, feature-by-feature
project for learning how modern RAG systems are designed, implemented, and evaluated.

## What this project is

- A **single Python project**, not a monorepo. Common files live at the root; each
  feature is a self-contained subpackage under `src/feature_*`.
- Learning-oriented: code favors **clarity over cleverness**. Files carry teaching
  comments explaining *why* (patterns, trade-offs), not just *what*.
- Each feature is a **continuation** of the previous one and reuses its code directly
  (e.g. Feature 2 imports Feature 1's ingestion instead of re-implementing chunking).

## Commands

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Gradio UI (one tab per feature)
python3 ui.py                # or: python3 main.py --ui

# CLI benchmarks
python3 main.py --list                                  # list registered features
python3 main.py --feature 1 --chunker all               # compare all 7 chunkers
python3 main.py --feature 1 --chunker recursive --show-chunks
python3 main.py --feature 2 --embedder langchain --no-metrics
python3 main.py --feature 2 --embedder all --metric cosine
python3 main.py --feature 2 --embedder hashing --store pgvector      # Postgres + pgvector
python3 main.py --feature 2 --embedder graph --store production      # graph + Neo4j + MinIO
```

Notes:
- No test suite or linter config yet — verify changes by running the relevant
  benchmark via `main.py` or the UI.
- `main.py`/`ui.py` add the repo root to `sys.path`, so imports use the `src.` prefix
  (e.g. `from src.lab import run_benchmark`). Run commands from the repo root.
- The OpenAI embedder needs `OPENAI_API_KEY` in the environment. When running
  `--embedder all` without it, `openai` is skipped automatically.
- Some embedders download models from Hugging Face on first use (semantic, splade,
  instruct, clip, code). `hashing` is dependency-light and fast — prefer it for quick checks.

## Architecture

```
rag-learning-lab/
├── main.py            # CLI benchmark dispatcher
├── ui.py              # Gradio UI, one tab per feature
├── docs/sample.txt    # shared source document for all features
└── src/
    ├── lab/                              # benchmark harness (shared)
    │   ├── registry.py                   # FeatureSpec + register/run_benchmark
    │   ├── types.py                      # BenchmarkReport / BenchmarkSection
    │   ├── config.py                     # DOC path, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
    │   ├── chunkers.py                   # shared chunker registry (CHUNKERS / CHUNKER_KEYS)
    │   └── benchmarks/                   # ONE module per feature (feature_1.py, feature_2.py)
    ├── feature_1_document_ingestion/     # load → clean → chunk (7 strategies)
    └── feature_2_embeddings_vector_store/# embed → store → top-k search (+ production tutorial)
```

### The benchmark harness (`src/lab`)

The lab is the glue that lets every feature be run and displayed uniformly:

- Each feature registers a `FeatureSpec` (id, name, description, `run` callable,
  `defaults` dict) via `register(...)` in `src/lab/benchmarks/feature_N.py`.
- Registration happens as an **import side-effect**: `src/lab/benchmarks/__init__.py`
  imports each feature module, and `main.py`/`ui.py` import `benchmarks` to trigger it.
- A feature's `run(...)` returns a `BenchmarkReport` — a list of `BenchmarkSection`s
  (title + markdown body). The same report renders to CLI (`report.to_cli()`) and to
  Gradio (`_report_to_sections`). Keep feature output as sections so both surfaces work.

### Feature packages (`src/feature_*`)

Each feature package re-exports its public API from its `__init__.py` so callers use
short imports (e.g. `from src.feature_2_embeddings_vector_store import build_index, search`).
Each has a `pipeline.py` that orchestrates the feature end-to-end and returns a
result dataclass (`IngestionResult`, `IndexResult`).

## Key conventions

- **Strategy Pattern everywhere.** Interchangeable components extend a `Base*` ABC and
  are selected via a registry dict keyed by a short CLI string:
  - Chunkers extend `BaseChunker` (`split(text) -> list[Chunk]`), registered in
    `src/lab/chunkers.py` (`CHUNKERS`).
  - Embedders extend `BaseEmbedder` (`embed(texts) -> np.ndarray`, `dimension`,
    `category`), registered in `src/lab/benchmarks/feature_2.py` (`EMBEDDERS`).
  - Vector stores extend `BaseVectorStore` (`add`, `search` returning best-first
    `SearchResult`s where higher `score` = more similar, regardless of metric),
    registered in `src/lab/benchmarks/feature_2.py` (`STORES`: `pgvector`,
    `production`). `pgvector` = Postgres + pgvector (LangChain);
    `production` = pgvector + Neo4j + MinIO tutorial over a research paper.
    Stores receive **precomputed vectors** from the embedder — they never re-embed.
- **Source / citations:** metadata (`source`, `chunk_index`, `char_start`,
  `char_end`) lives on every `VectorRecord`; render it with `format_citation(...)`
  rather than hand-formatting per store, so all backends cite identically.
- **Adding a new strategy:** create the class under the right subpackage, re-export it
  from the package `__init__.py`, then add an entry to the relevant registry dict (and a
  category/tag if the pattern has one, e.g. `EMBEDDER_CATEGORIES`, `EMBEDDER_TAGS`).
- **Adding a new feature:** create `src/feature_N_*/` with a `pipeline.py` and public
  `__init__.py`; add `src/lab/benchmarks/feature_N.py` that builds a `BenchmarkReport`
  and calls `register(FeatureSpec(...))`; import it in `src/lab/benchmarks/__init__.py`;
  wire CLI args in `main.py` and a tab in `ui.py`; update `README.md`'s learning path.
- **Style:** `from __future__ import annotations` and modern type hints; dataclasses for
  structured results; NumPy `float32` vectors of shape `(n_texts, dimension)`. Comments
  explain intent/trade-offs, not obvious mechanics.
- Shared defaults (chunk size, overlap, top-k, source doc) live in `src/lab/config.py` —
  reuse them rather than hardcoding.

## Feature status

- Feature 1 — Document Ingestion + Chunking ✅
- Feature 2 — Embeddings + Vector Store ✅
- Features 3–9 (retrieval/answering, agent harness, memory, tracing, eval, release) —
  planned. See `README.md` for the full roadmap.

## Per-feature docs

Each feature has its own README with deeper detail:
- `src/feature_1_document_ingestion/README.md`
- `src/feature_2_embeddings_vector_store/README.md`
