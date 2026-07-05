#!/usr/bin/env python3
"""Minimal CLI — dispatch feature benchmarks. Use ui.py for the visual lab."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.lab import list_features, run_benchmark  # noqa: E402
from src.lab import benchmarks  # noqa: E402, F401 — register features
from src.lab.chunkers import CHUNKER_KEYS  # noqa: E402
from src.lab.benchmarks.feature_2 import EMBEDDERS, METRICS  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    features = list_features()

    p = argparse.ArgumentParser(description="RAG Learning Lab — feature benchmarks")
    p.add_argument("--ui", action="store_true", help="Launch Gradio UI")
    p.add_argument("--feature", type=int, choices=[f.id for f in features], help="Feature to benchmark")
    p.add_argument(
        "--chunker",
        choices=list(CHUNKER_KEYS) + ["all"],
        help="Chunker key (feature 1: all|key; feature 2: any key)",
    )
    p.add_argument("--embedder", choices=list(EMBEDDERS) + ["all"], help="Feature 2 only")
    p.add_argument("--metric", choices=list(METRICS), default="cosine", help="Feature 2 only")
    p.add_argument("--show-chunks", action="store_true", help="Feature 1: include chunk text")
    p.add_argument("--no-metrics", action="store_true", help="Feature 2: skip metric comparison")
    p.add_argument("--list", action="store_true", help="List registered features")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.list:
        for f in list_features():
            print(f"  {f.id}. {f.name} — {f.description}")
        return

    if args.ui:
        from ui import launch

        launch()
        return

    if args.feature is None:
        print("Pick a feature: --feature 1  or  --feature 2  (or --ui)")
        print("Use --list to see all features.")
        return

    opts: dict = {}
    if args.feature == 1:
        opts["chunker"] = args.chunker or "all"
        opts["show_chunks"] = args.show_chunks
    else:
        opts["chunker"] = args.chunker or "recursive"
        if opts["chunker"] == "all":
            opts["chunker"] = "recursive"
        opts["embedder"] = args.embedder or "all"
        opts["metric"] = args.metric
        opts["show_metrics"] = not args.no_metrics

    report = run_benchmark(args.feature, **opts)
    print(report.to_cli())


if __name__ == "__main__":
    main()
