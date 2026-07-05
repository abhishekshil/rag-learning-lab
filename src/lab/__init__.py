# Lab benchmark harness — register features in lab/benchmarks/, run via main.py or ui.py.

from .config import DOC, ROOT
from .registry import get_feature, list_features, run_benchmark
from .types import BenchmarkReport, BenchmarkSection

# Side-effect: register all feature benchmarks.
from . import benchmarks  # noqa: F401

__all__ = [
    "BenchmarkReport",
    "BenchmarkSection",
    "DOC",
    "ROOT",
    "get_feature",
    "list_features",
    "run_benchmark",
]
