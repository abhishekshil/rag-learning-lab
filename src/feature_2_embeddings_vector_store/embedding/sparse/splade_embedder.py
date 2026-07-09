# splade_embedder.py
# Category: SPARSE — learned sparse embeddings (SPLADE via sentence-transformers).
#
# Smarter than hashing: neural model expands important terms while staying sparse.
# We densify vectors so the pgvector store can index them unchanged.
#
# Note: SPLADE has no dedicated langchain-huggingface class yet; sentence-transformers
# SparseEncoder is the standard path and matches LangChain's recommended approach.

from __future__ import annotations

import numpy as np

from ..base_embedder import BaseEmbedder


class SpladeEmbedder(BaseEmbedder):
    category = "sparse"

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        from sentence_transformers import SparseEncoder

        self.model_name = model_name
        # CPU avoids MPS sparse-tensor errors on Apple Silicon.
        self._model = SparseEncoder(model_name, device="cpu")
        probe = self._to_dense(
            self._model.encode_document(["probe"], convert_to_tensor=True)
        )
        self._dimension = int(probe.shape[-1])

    @staticmethod
    def _to_dense(vectors) -> np.ndarray:
        import torch

        if isinstance(vectors, torch.Tensor):
            if vectors.layout == torch.sparse_coo:
                vectors = vectors.to_dense()
            return vectors.detach().cpu().numpy().astype(np.float32)
        return np.asarray(vectors, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode_document(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return self._normalize(self._to_dense(vectors))

    def embed_one(self, text: str) -> np.ndarray:
        vectors = self._model.encode_query(
            [text],
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return self._normalize(self._to_dense(vectors))[0]

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return (vectors / norms).astype(np.float32)

    def name(self) -> str:
        short = self.model_name.split("/")[-1]
        return f"SPLADE({short}, dim={self._dimension}, sparse/learned)"
