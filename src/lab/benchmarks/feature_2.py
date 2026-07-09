# Feature 2 benchmark — embeddings, vector store, and retrieval comparison.

from __future__ import annotations

import os

from ..chunkers import CHUNKER_KEYS, get_chunker
from ..config import DOC, TOP_K
from ..registry import FeatureSpec, register
from ..types import BenchmarkReport, BenchmarkSection
from ...feature_1_document_ingestion import run_ingestion
from ...feature_2_embeddings_vector_store import (
    CATEGORY_LABELS,
    EMBEDDER_CATEGORIES,
    ClipEmbedder,
    CodeEmbedder,
    HashingEmbedder,
    HybridEmbedder,
    InstructEmbedder,
    GraphEmbedder,
    LangChainSentenceTransformerEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    SpladeEmbedder,
    build_index,
    build_production_index,
    chunks_to_records,
    fetch_raw_paper,
    search,
    search_with_graph_expansion,
    format_citation,
    PgVectorStore,
    Neo4jCitationGraph,
    MinioBlobStore,
    PAPER_PATH,
    DEFAULT_PAPER,
)

EMBEDDERS = {
    "hashing": lambda: HashingEmbedder(dimension=512),
    "splade": lambda: SpladeEmbedder(),
    "semantic": lambda: SentenceTransformerEmbedder(),
    "langchain": lambda: LangChainSentenceTransformerEmbedder(),
    "graph": lambda: GraphEmbedder(),
    "hybrid": lambda: HybridEmbedder(),
    "instruct": lambda: InstructEmbedder(),
    "clip": lambda: ClipEmbedder(),
    "code": lambda: CodeEmbedder(),
    "openai": lambda: OpenAIEmbedder(),
}

EMBEDDER_TAGS = {
    "hashing": "SPARSE",
    "splade": "SPARSE (learned)",
    "semantic": "DENSE",
    "langchain": "DENSE (LangChain)",
    "graph": "GRAPH (LangChain + citations)",
    "hybrid": "HYBRID",
    "instruct": "INSTRUCT",
    "clip": "MULTIMODAL",
    "code": "CODE",
    "openai": "API",
}

OPTIONAL_EMBEDDERS = {"openai"}
METRICS = ("cosine", "dot", "euclidean")

# pgvector   — Postgres + pgvector via LangChain (needs docker compose)
# production — pgvector + Neo4j + MinIO tutorial stack (needs docker compose)
STORES = ("pgvector", "production")


def _make_store(store_kind: str, metric: str, namespace: str, embedder=None):
    from ...feature_2_embeddings_vector_store.embedding.langchain_adapter import (
        default_langchain_embeddings,
        embedder_as_langchain,
    )

    lc_embeddings = (
        embedder_as_langchain(embedder) if embedder else default_langchain_embeddings()
    )
    return PgVectorStore(
        metric=metric,
        collection_name=f"f2_{namespace}",
        embeddings=lc_embeddings,
        reset=True,
    )


QUERIES = [
    "What improved on FlashAttention?",
    "How does FlashAttention reduce memory reads between HBM and SRAM?",
    "What is the relationship between FlashAttention and the Transformer?",
]

PRODUCTION_QUERIES = [
    "What improved on FlashAttention?",
    "IO-aware tiling for GPU attention",
    "FlashAttention-2 speedup over original",
]


def _catalog_markdown() -> str:
    lines = []
    for category, keys in EMBEDDER_CATEGORIES.items():
        lines.append(f"**{category.upper()}** — {CATEGORY_LABELS[category]}")
        for key in keys:
            lines.append(f"- `{key}`")
        lines.append("")
    return "\n".join(lines)


def _resolve_embedders(choice: str) -> list[str]:
    if choice != "all":
        return [choice]
    names = list(EMBEDDERS)
    if "openai" in names and not os.environ.get("OPENAI_API_KEY"):
        names = [n for n in names if n not in OPTIONAL_EMBEDDERS]
    return names


def _index_table(index_results) -> str:
    lines = [
        "| Embedder | Dim | Chunks | Embed (s) |",
        "|----------|-----|--------|-----------|",
    ]
    for r in index_results:
        lines.append(
            f"| {r.embedder_name} | {r.dimension} | {r.chunk_count} | {r.embed_seconds:.3f} |"
        )
    return "\n".join(lines)


def _preview(text: str, width: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _query_results_markdown(query: str, labelled: list[tuple[str, list]]) -> str:
    lines = [f"### {query}", ""]
    for label, hits in labelled:
        lines.append(f"**{label}**")
        if not hits:
            lines.append("- *(no results)*")
        else:
            for rank, res in enumerate(hits, start=1):
                lines.append(
                    f"{rank}. `score={res.score:+.3f}` "
                    f"`{res.record.id}`  \n"
                    f"   ↳ source: *{format_citation(res)}*  \n"
                    f"   {_preview(res.record.text)}"
                )
        lines.append("")
    return "\n".join(lines)


def _graph_hits_markdown(hits) -> str:
    if not hits:
        return "*No graph neighbors found.*"
    lines = []
    for h in hits:
        chunks = ", ".join(h.chunk_ids) if h.chunk_ids else "—"
        lines.append(f"- `{h.name}` —[{h.relation}]→ `{h.neighbor}` (chunks: {chunks})")
    return "\n".join(lines)


def _run_production_benchmark(
    chunker: str,
    embedder: str,
    metric: str,
    show_metrics: bool,
) -> BenchmarkReport:
    """Tutorial over FlashAttention paper: MinIO + Neo4j + pgvector."""
    embedder_names = _resolve_embedders("langchain" if embedder == "all" else embedder)
    chunker_instance = get_chunker(chunker)
    name = embedder_names[0]
    emb = EMBEDDERS[name]()

    try:
        prod = build_production_index(DEFAULT_PAPER, chunker_instance, emb, metric=metric, reset=True)
    except Exception as exc:
        return BenchmarkReport(
            feature_id=2,
            feature_name="Embeddings + Vector Store",
            subtitle="production stack (docker required)",
            sections=[
                BenchmarkSection(
                    title="Docker required",
                    body=(
                        f"Could not connect to the production stack: `{exc}`\n\n"
                        "**Start services:**\n"
                        "```bash\n"
                        "docker compose up -d\n"
                        "```\n"
                        "Then re-run with `--store production`."
                    ),
                )
            ],
            meta={"store": "production", "error": str(exc)},
        )

    graph = Neo4jCitationGraph()
    sections = [
        BenchmarkSection(
            title="Production storage tutorial",
            body=(
                f"Based on [production-rag-storage-choices.md](docs/production-rag-storage-choices.md).\n\n"
                f"**Paper:** `{DEFAULT_PAPER}`  \n"
                f"**Chunker:** {chunker} (`{chunker_instance.strategy_name()}`)  \n"
                f"**Embedder:** {name}  \n"
                f"**Blob pointer:** `{prod.media_uri}`  \n"
                f"**Graph edges seeded:** {prod.graph_edges}  \n"
                f"**Chunks indexed:** {prod.index.chunk_count}"
            ),
        ),
        BenchmarkSection(
            title="Three storage layers",
            body=(
                "| Layer | Technology | What it stores |\n"
                "|-------|------------|----------------|\n"
                f"| Vector | {prod.index.store.name()} | embeddings + text + metadata |\n"
                f"| Graph | {prod.graph_store_name} | citation edges (improves_on, builds_on) |\n"
                f"| Blob | {prod.blob_store_name} | raw paper file (s3:// pointer in metadata) |"
            ),
            kind="table",
        ),
        BenchmarkSection(
            title="Index build",
            body=_index_table([prod.index]),
            kind="table",
        ),
    ]

    store = prod.index.store
    for query in PRODUCTION_QUERIES:
        vector_hits, graph_hits = search_with_graph_expansion(
            store, graph, emb, query, k=TOP_K, graph_seed="FlashAttention"
        )
        body = _query_results_markdown(query, [(f"pgvector | {emb.name()}", vector_hits)])
        body += "\n**Graph expansion (1-hop from FlashAttention):**\n\n"
        body += _graph_hits_markdown(graph_hits)
        sections.append(BenchmarkSection(title="Query", body=body))

    # Demo: fetch raw file from MinIO via pointer
    try:
        blob = MinioBlobStore()
        snippet = fetch_raw_paper(blob, prod.media_uri)[:200]
        sections.append(
            BenchmarkSection(
                title="Blob fetch demo",
                body=(
                    f"Fetched first 200 chars from `{prod.media_uri}`:\n\n"
                    f"> {_preview(snippet, 200)}"
                ),
            )
        )
    except Exception as exc:
        sections.append(
            BenchmarkSection(title="Blob fetch demo", body=f"*(skipped: {exc})*")
        )

    return BenchmarkReport(
        feature_id=2,
        feature_name="Embeddings + Vector Store",
        subtitle=f"production tutorial  chunker={chunker}  embedder={name}  metric={metric}",
        sections=sections,
        meta={"embedders": embedder_names, "metric": metric, "store": "production"},
    )


def run(
    chunker: str = "recursive",
    embedder: str = "all",
    metric: str = "cosine",
    store: str = "pgvector",
    show_metrics: bool = True,
) -> BenchmarkReport:
    store_kind = store if store in STORES else "pgvector"

    if store_kind == "production":
        return _run_production_benchmark(chunker, embedder, metric, show_metrics)

    embedder_names = _resolve_embedders(embedder)
    chunker_instance = get_chunker(chunker)
    source_doc = DOC

    index_results = []
    stores: list[tuple[str, object, object]] = []
    skipped = []
    if embedder == "all" and "openai" not in embedder_names:
        skipped.append("openai (set OPENAI_API_KEY)")

    for name in embedder_names:
        emb = EMBEDDERS[name]()
        vector_store = _make_store(store_kind, metric, namespace=f"{name}_{metric}", embedder=emb)
        index_results.append(build_index(source_doc, get_chunker(chunker), emb, vector_store))
        stores.append((name, emb, vector_store))

    store_note = " (Postgres + pgvector via LangChain — run `docker compose up -d`)"

    sections = [
        BenchmarkSection(
            title="Overview",
            body=(
                f"**Source:** `{source_doc}`  \n"
                f"**Chunker:** {chunker} (`{chunker_instance.strategy_name()}`)  \n"
                f"**Store:** {store_kind}{store_note}  \n"
                f"**Metric:** {metric}  \n"
                f"**Embedders:** {', '.join(embedder_names)}"
                + (f"  \n**Skipped:** {', '.join(skipped)}" if skipped else "")
                + (
                    f"\n\n*For the FlashAttention paper tutorial use "
                    f"`--store production` (paper at `{PAPER_PATH}`).*"
                )
            ),
        ),
        BenchmarkSection(title="Embedder catalog", body=_catalog_markdown()),
        BenchmarkSection(title="Index build", body=_index_table(index_results), kind="table"),
    ]

    queries = QUERIES
    for query in queries:
        labelled = []
        for name, emb, vec_store in stores:
            tag = EMBEDDER_TAGS.get(name, name.upper())
            labelled.append((f"{tag} | {emb.name()}", search(vec_store, emb, query, k=TOP_K)))
        sections.append(BenchmarkSection(title="Query", body=_query_results_markdown(query, labelled)))

    if show_metrics and "langchain" in embedder_names:
        emb = EMBEDDERS["langchain"]()
        ingestion = run_ingestion(source_doc, get_chunker(chunker))
        records, _ = chunks_to_records(ingestion.chunks, emb, source_doc)
        labelled = []
        for m in METRICS:
            metric_store = _make_store(store_kind, m, namespace=f"langchain_metric_{m}", embedder=emb)
            metric_store.add(records)
            labelled.append((f"metric = {m}", search(metric_store, emb, QUERIES[2], k=TOP_K)))
        sections.append(
            BenchmarkSection(
                title="Metric comparison",
                body=_query_results_markdown(QUERIES[2], labelled)
                + "\n\n*With unit-normalized vectors, cosine and dot rank identically.*",
            )
        )

    return BenchmarkReport(
        feature_id=2,
        feature_name="Embeddings + Vector Store",
        subtitle=f"chunker={chunker}  embedder={embedder}  store={store_kind}  metric={metric}",
        sections=sections,
        meta={"embedders": embedder_names, "metric": metric, "store": store_kind},
    )


register(
    FeatureSpec(
        id=2,
        name="Embeddings + Vector Store",
        description="Embed chunks, build indexes, and compare retrieval.",
        run=run,
        defaults={
            "chunker": "recursive",
            "embedder": "langchain",
            "metric": "cosine",
            "store": "pgvector",
            "show_metrics": True,
        },
    )
)
