# Feature 1 benchmark — document ingestion + chunking comparison.

from __future__ import annotations

from ..chunkers import CHUNKERS, CHUNKER_KEYS, get_chunker
from ..config import CHUNK_SIZE, DOC
from ..registry import FeatureSpec, register
from ..types import BenchmarkReport, BenchmarkSection
from ...feature_1_document_ingestion import run_ingestion


def _comparison_table(results) -> str:
    lines = [
        "| Strategy | Chunks | Avg chars | Min | Max |",
        "|----------|--------|-----------|-----|-----|",
    ]
    for r in results:
        lines.append(
            f"| {r.strategy_name} | {r.chunk_count} | {r.avg_chunk_size:.0f} "
            f"| {r.min_chunk_size} | {r.max_chunk_size} |"
        )
    return "\n".join(lines)


def _chunk_detail(result) -> str:
    lines = [
        f"**Raw chars:** {result.raw_char_count}  ",
        f"**Clean chars:** {result.clean_char_count}  ",
        f"**Total chunks:** {result.chunk_count}",
        "",
    ]
    for chunk in result.chunks:
        lines.append(f"### Chunk {chunk.index} (chars {chunk.char_start}–{chunk.char_end})")
        lines.append(f"```\n{chunk.text}\n```")
        lines.append("")
    return "\n".join(lines)


def run(chunker: str = "all", show_chunks: bool = False) -> BenchmarkReport:
    names = list(CHUNKERS) if chunker == "all" else [chunker]
    results = []
    for name in names:
        results.append(run_ingestion(DOC, CHUNKERS[name]()))

    sections = [
        BenchmarkSection(
            title="Overview",
            body=(
                f"**Source:** `{DOC}`  \n"
                f"**Strategies run:** {len(results)}  \n"
                f"**Chunk size target:** {CHUNK_SIZE} chars"
            ),
        ),
        BenchmarkSection(title="Strategy comparison", body=_comparison_table(results), kind="table"),
    ]

    if show_chunks:
        for result in results:
            sections.append(
                BenchmarkSection(
                    title=f"Chunks — {result.strategy_name}",
                    body=_chunk_detail(result),
                    kind="code",
                )
            )

    return BenchmarkReport(
        feature_id=1,
        feature_name="Document Ingestion + Chunking",
        subtitle=f"chunker={chunker}",
        sections=sections,
        meta={"chunker": chunker, "strategies": [r.strategy_name for r in results]},
    )


register(
    FeatureSpec(
        id=1,
        name="Document Ingestion + Chunking",
        description="Load, clean, and compare chunking strategies.",
        run=run,
        defaults={"chunker": "all", "show_chunks": False},
    )
)
