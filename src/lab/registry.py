# Feature benchmark registry — add new features by registering a runner here.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .types import BenchmarkReport

Runner = Callable[..., BenchmarkReport]


@dataclass(frozen=True)
class FeatureSpec:
    id: int
    name: str
    description: str
    run: Runner
    defaults: dict[str, Any]


_REGISTRY: dict[int, FeatureSpec] = {}


def register(spec: FeatureSpec) -> None:
    _REGISTRY[spec.id] = spec


def list_features() -> list[FeatureSpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def get_feature(feature_id: int) -> FeatureSpec:
    if feature_id not in _REGISTRY:
        known = ", ".join(str(k) for k in sorted(_REGISTRY))
        raise KeyError(f"Unknown feature {feature_id}. Known: {known}")
    return _REGISTRY[feature_id]


def run_benchmark(feature_id: int, **options: Any) -> BenchmarkReport:
    spec = get_feature(feature_id)
    merged = {**spec.defaults, **{k: v for k, v in options.items() if v is not None}}
    return spec.run(**merged)
