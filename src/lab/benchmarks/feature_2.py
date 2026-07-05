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
    LangChainSentenceTransformerEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    SpladeEmbedder,
    build_index,
    chunks_to_records,
    search,
    InMemoryVectorStore,
)

EMBEDDERS = {
    "hashing": lambda: HashingEmbedder(dimension=512),
    "splade": lambda: SpladeEmbedder(),
    "semantic": lambda: SentenceTransformerEmbedder(),
    "langchain": lambda: LangChainSentenceTransformerEmbedder(),
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
    "hybrid": "HYBRID",
    "instruct": "INSTRUCT",
    "clip": "MULTIMODAL",
    "code": "CODE",
    "openai": "API",
}

OPTIONAL_EMBEDDERS = {"openai"}
METRICS = ("cosine", "dot", "euclidean")

QUERIES = [
    "How can a model answer questions about recent events it was never trained on?",
    "why repeat a little text between neighboring pieces",
    "how does the system find passages related to a user's question?",
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
                meta = res.record.metadata
                lines.append(
                    f"{rank}. `score={res.score:+.3f}` "
                    f"`{res.record.id}` chars {meta['char_start']}–{meta['char_end']}  \n"
                    f"   {_preview(res.record.text)}"
                )
        lines.append("")
    return "\n".join(lines)


def run(
    chunker: str = "recursive",
    embedder: str = "all",
    metric: str = "cosine",
    show_metrics: bool = True,
) -> BenchmarkReport:
    embedder_names = _resolve_embedders(embedder)
    chunker_instance = get_chunker(chunker)

    index_results = []
    stores: list[tuple[str, object, object]] = []
    skipped = []
    if embedder == "all" and "openai" not in embedder_names:
        skipped.append("openai (set OPENAI_API_KEY)")

    for name in embedder_names:
        emb = EMBEDDERS[name]()
        store = InMemoryVectorStore(metric=metric)
        index_results.append(build_index(DOC, get_chunker(chunker), emb, store))
        stores.append((name, emb, store))

    sections = [
        BenchmarkSection(
            title="Overview",
            body=(
                f"**Source:** `{DOC}`  \n"
                f"**Chunker:** {chunker} (`{chunker_instance.strategy_name()}`)  \n"
                f"**Metric:** {metric}  \n"
                f"**Embedders:** {', '.join(embedder_names)}"
                + (f"  \n**Skipped:** {', '.join(skipped)}" if skipped else "")
            ),
        ),
        BenchmarkSection(title="Embedder catalog", body=_catalog_markdown()),
        BenchmarkSection(title="Index build", body=_index_table(index_results), kind="table"),
    ]

    for query in QUERIES:
        labelled = []
        for name, emb, store in stores:
            tag = EMBEDDER_TAGS.get(name, name.upper())
            labelled.append((f"{tag} | {emb.name()}", search(store, emb, query, k=TOP_K)))
        sections.append(BenchmarkSection(title="Query", body=_query_results_markdown(query, labelled)))

    if show_metrics and "semantic" in embedder_names:
        emb = EMBEDDERS["semantic"]()
        ingestion = run_ingestion(DOC, get_chunker(chunker))
        records, _ = chunks_to_records(ingestion.chunks, emb, DOC)
        labelled = []
        for m in METRICS:
            store = InMemoryVectorStore(metric=m)
            store.add(records)
            labelled.append((f"metric = {m}", search(store, emb, QUERIES[2], k=TOP_K)))
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
        subtitle=f"chunker={chunker}  embedder={embedder}  metric={metric}",
        sections=sections,
        meta={"embedders": embedder_names, "metric": metric},
    )


register(
    FeatureSpec(
        id=2,
        name="Embeddings + Vector Store",
        description="Embed chunks, build indexes, and compare retrieval.",
        run=run,
        defaults={
            "chunker": "recursive",
            "embedder": "all",
            "metric": "cosine",
            "show_metrics": True,
        },
    )
)
