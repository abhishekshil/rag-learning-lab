# instruct_embedder.py
# Category: INSTRUCT — query-aware dense embeddings (E5-style prefixes).
#
# Uses LangChain HuggingFaceEmbeddings with different prefixes:
#   - documents/chunks: "passage: {text}"
#   - search queries:   "query: {text}"
#
# Model: multilingual-e5-base (768d) — works for English and many other languages.

from __future__ import annotations

import numpy as np

from ..base_embedder import BaseEmbedder


class InstructEmbedder(BaseEmbedder):
    category = "instruct"

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
    ):
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._dimension = len(
            self._embeddings.embed_query(f"{query_prefix}probe")
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"{self._document_prefix}{text}" for text in texts]
        vectors = self._embeddings.embed_documents(prefixed)
        return np.array(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        vector = self._embeddings.embed_query(f"{self._query_prefix}{text}")
        return np.array(vector, dtype=np.float32)

    def name(self) -> str:
        short = self.model_name.split("/")[-1]
        return f"InstructE5({short}, dim={self._dimension})"
