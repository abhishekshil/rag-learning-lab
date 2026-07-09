#!/usr/bin/env python3
"""Gradio lab UI — run feature benchmarks and view results by section."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import gradio as gr

from src.lab import list_features, run_benchmark
from src.lab import benchmarks  # noqa: F401 — register features
from src.lab.chunkers import CHUNKER_KEYS
from src.lab.benchmarks.feature_2 import EMBEDDERS, METRICS, STORES


def _report_to_sections(report) -> list[tuple[str, str]]:
    return [(s.title, s.body) for s in report.sections]


def _run_feature_1(chunker: str, show_chunks: bool):
    report = run_benchmark(1, chunker=chunker, show_chunks=show_chunks)
    sections = _report_to_sections(report)
    header = f"## Feature 1: {report.feature_name}\n*{report.subtitle}*"
    return header, sections


def _run_feature_2(chunker: str, embedder: str, metric: str, store: str, show_metrics: bool):
    report = run_benchmark(
        2,
        chunker=chunker,
        embedder=embedder,
        metric=metric,
        store=store,
        show_metrics=show_metrics,
    )
    sections = _report_to_sections(report)
    header = f"## Feature 2: {report.feature_name}\n*{report.subtitle}*"
    return header, sections


def _render_sections(header: str, sections: list[tuple[str, str]]) -> str:
    """Single markdown blob with clear section breaks for Gradio."""
    if not sections:
        return header + "\n\n*No sections yet — click Run benchmark.*"
    parts = [header, ""]
    for title, body in sections:
        parts.append(f"---\n### {title}\n\n{body}")
    return "\n\n".join(parts)


def launch() -> None:
    feature_list = "\n".join(f"- **{f.id}.** {f.name} — {f.description}" for f in list_features())

    with gr.Blocks(title="RAG Learning Lab", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# RAG Learning Lab\n"
            "Benchmark each feature in isolation. Results appear in **sections** below.\n\n"
            f"{feature_list}"
        )

        with gr.Tabs():
            # ── Feature 1 ──────────────────────────────────────────────────
            with gr.Tab("① Chunking"):
                with gr.Row():
                    f1_chunker = gr.Dropdown(
                        choices=["all"] + list(CHUNKER_KEYS),
                        value="all",
                        label="Chunker",
                    )
                    f1_show = gr.Checkbox(label="Show chunk text", value=False)
                f1_run = gr.Button("Run benchmark", variant="primary")
                f1_out = gr.Markdown("*Ready.*")

                def on_f1(chunker, show):
                    h, secs = _run_feature_1(chunker, show)
                    return _render_sections(h, secs)

                f1_run.click(on_f1, [f1_chunker, f1_show], f1_out)

            # ── Feature 2 ──────────────────────────────────────────────────
            with gr.Tab("② Embeddings"):
                with gr.Row():
                    f2_chunker = gr.Dropdown(
                        choices=list(CHUNKER_KEYS),
                        value="recursive",
                        label="Chunker (from Feature 1)",
                        info="Any chunker × any embedder",
                    )
                    f2_embedder = gr.Dropdown(
                        choices=["all"] + list(EMBEDDERS),
                        value="langchain",
                        label="Embedder",
                        info="langchain (default) · graph fuses citations · 'all' is slow",
                    )
                with gr.Row():
                    f2_metric = gr.Dropdown(choices=list(METRICS), value="cosine", label="Metric")
                    f2_store = gr.Dropdown(
                        choices=list(STORES),
                        value="pgvector",
                        label="Vector store",
                        info="requires: docker compose up -d",
                    )
                    f2_metrics = gr.Checkbox(label="Metric comparison", value=False)
                f2_run = gr.Button("Run benchmark", variant="primary")
                f2_out = gr.Markdown("*Ready. Default: langchain embedder. Try `graph` with `--store production`.*")

                def on_f2(chunker, embedder, metric, store, show_metrics):
                    h, secs = _run_feature_2(chunker, embedder, metric, store, show_metrics)
                    return _render_sections(h, secs)

                f2_run.click(
                    on_f2,
                    [f2_chunker, f2_embedder, f2_metric, f2_store, f2_metrics],
                    f2_out,
                )

        gr.Markdown(
            "---\n"
            "**CLI:** `python3 main.py --feature 1 --chunker all`  \n"
            "**API:** OpenAI embedder needs `OPENAI_API_KEY` in the environment."
        )

    app.launch()


if __name__ == "__main__":
    launch()
