# hybrid_embedder.py
# Category: HYBRID — fuses sparse (hashing) + dense (semantic) in one vector.

from typing import Optional

import numpy as np

from ..base_embedder import BaseEmbedder
from ..dense import SentenceTransformerEmbedder
from ..sparse import HashingEmbedder


class HybridEmbedder(BaseEmbedder):
    category = "hybrid"

    def __init__(
        self,
        sparse: Optional[HashingEmbedder] = None,
        dense: Optional[SentenceTransformerEmbedder] = None,
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7,
    ):
        if sparse_weight < 0 or dense_weight < 0:
            raise ValueError("weights must be non-negative")
        if sparse_weight == 0 and dense_weight == 0:
            raise ValueError("at least one weight must be positive")

        self._sparse = sparse or HashingEmbedder()
        self._dense = dense or SentenceTransformerEmbedder()
        self._sparse_weight = sparse_weight
        self._dense_weight = dense_weight

    @property
    def sparse(self) -> HashingEmbedder:
        return self._sparse

    @property
    def dense(self) -> SentenceTransformerEmbedder:
        return self._dense

    @property
    def dimension(self) -> int:
        return self._sparse.dimension + self._dense.dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        sparse_vectors = self._sparse.embed(texts)
        dense_vectors = self._dense.embed(texts)

        combined = np.hstack(
            [
                sparse_vectors * self._sparse_weight,
                dense_vectors * self._dense_weight,
            ]
        )

        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return (combined / norms).astype(np.float32)

    def name(self) -> str:
        return (
            f"Hybrid(sparse={self._sparse_weight:g}, dense={self._dense_weight:g}, "
            f"dim={self.dimension})"
        )
