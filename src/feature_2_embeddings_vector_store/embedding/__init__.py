# Embedding strategies for Feature 2, organized by category.
#
#   sparse     — hashing, splade
#   dense      — semantic (direct ST), langchain (ST via LangChain)
#   hybrid     — hashing + semantic fused
#   instruct   — E5 query/passage prefixes
#   multimodal — CLIP text tower
#   code       — code-tuned model
#   api        — OpenAI cloud embeddings

from .base_embedder import BaseEmbedder
from .categories import ALL_EMBEDDER_KEYS, CATEGORY_LABELS, EMBEDDER_CATEGORIES
from .sparse import HashingEmbedder, SpladeEmbedder
from .dense import LangChainSentenceTransformerEmbedder, SentenceTransformerEmbedder
from .hybrid import HybridEmbedder
from .instruct import InstructEmbedder
from .multimodal import ClipEmbedder
from .code import CodeEmbedder
from .api import OpenAIEmbedder

__all__ = [
    "BaseEmbedder",
    "ALL_EMBEDDER_KEYS",
    "CATEGORY_LABELS",
    "EMBEDDER_CATEGORIES",
    "HashingEmbedder",
    "SpladeEmbedder",
    "SentenceTransformerEmbedder",
    "LangChainSentenceTransformerEmbedder",
    "HybridEmbedder",
    "InstructEmbedder",
    "ClipEmbedder",
    "CodeEmbedder",
    "OpenAIEmbedder",
]
