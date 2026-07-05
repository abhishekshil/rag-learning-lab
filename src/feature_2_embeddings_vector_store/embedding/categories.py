# categories.py
# Taxonomy for all embedding strategies in Feature 2.
#
# Categories map to common RAG retrieval patterns (no fine-tuning).

from __future__ import annotations

from typing import Literal

EmbedderCategory = Literal[
    "sparse",
    "dense",
    "hybrid",
    "instruct",
    "multimodal",
    "code",
    "api",
]

CATEGORY_LABELS: dict[EmbedderCategory, str] = {
    "sparse": "Sparse — keyword / lexical matching",
    "dense": "Dense — general semantic bi-encoder",
    "hybrid": "Hybrid — sparse + dense fused in one vector",
    "instruct": "Instruct — query-aware (different hats for Q vs doc)",
    "multimodal": "Multimodal — text + image in one space",
    "code": "Code — source code and technical text",
    "api": "API — cloud-hosted embeddings",
}

# Registry keys (CLI --embedder values) grouped by category.
EMBEDDER_CATEGORIES: dict[EmbedderCategory, tuple[str, ...]] = {
    "sparse": ("hashing", "splade"),
    "dense": ("semantic", "langchain"),
    "hybrid": ("hybrid",),
    "instruct": ("instruct",),
    "multimodal": ("clip",),
    "code": ("code",),
    "api": ("openai",),
}

ALL_EMBEDDER_KEYS: tuple[str, ...] = tuple(
    key for keys in EMBEDDER_CATEGORIES.values() for key in keys
)
