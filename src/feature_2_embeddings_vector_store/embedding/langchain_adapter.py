# langchain_adapter.py
# Shared bridge: LangChain Embeddings → our BaseEmbedder (NumPy vectors).
#
# LangChain uses embed_documents() for chunks and embed_query() for search.
# Symmetric models treat both the same; instruct models override prefixes
# in their own classes but still call this adapter underneath.

from __future__ import annotations

import numpy as np

from .base_embedder import BaseEmbedder


class LangChainEmbedder(BaseEmbedder):
    """Wrap any LangChain Embeddings instance as a BaseEmbedder."""

    category: str = "dense"

    def __init__(
        self,
        embeddings,
        *,
        display_name: str,
        category: str = "dense",
    ):
        self._embeddings = embeddings
        self._display_name = display_name
        self.category = category
        self._dimension = len(self._embeddings.embed_query("probe"))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._embeddings.embed_documents(texts)
        return np.array(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        vector = self._embeddings.embed_query(text)
        return np.array(vector, dtype=np.float32)

    def name(self) -> str:
        return f"{self._display_name}(dim={self._dimension})"
