# in_memory_store.py
# A brute-force, in-memory vector store. Exact and simple: it compares the
# query against EVERY stored vector. Perfect for learning and for small data.
# (Production systems swap this for an approximate index like HNSW/IVF when
# there are millions of vectors, but the interface stays the same.)
#
# Supported similarity metrics:
#   - "cosine":    angle between vectors, ignores length. Best default for text.
#   - "dot":       dot product. Rewards alignment AND magnitude.
#   - "euclidean": straight-line distance (smaller = closer). We negate it so
#                  that, like the others, a HIGHER score means more similar.
#
# Note: when vectors are already unit-normalized (our embedders do this),
# cosine and dot product produce the same ranking.

from __future__ import annotations

import numpy as np

from .base_store import BaseVectorStore, VectorRecord, SearchResult

_METRICS = ("cosine", "dot", "euclidean")


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, metric: str = "cosine"):
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got '{metric}'")
        self.metric = metric
        self._records: list[VectorRecord] = []
        self._matrix: np.ndarray | None = None  # shape (n, dim), rebuilt on add

    def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._records.extend(records)
        # Rebuild the stacked matrix so search is one vectorized NumPy op.
        self._matrix = np.vstack([r.vector for r in self._records]).astype(np.float32)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[SearchResult]:
        if self._matrix is None or len(self._records) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self._score_all(query)

        k = min(k, len(self._records))
        # argpartition gets the top-k cheaply, then we sort just those k.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            SearchResult(record=self._records[i], score=float(scores[i]))
            for i in top_idx
        ]

    def _score_all(self, query: np.ndarray) -> np.ndarray:
        """Return a similarity score per stored vector (higher = more similar)."""
        matrix = self._matrix

        if self.metric == "dot":
            return matrix @ query

        if self.metric == "cosine":
            q_norm = np.linalg.norm(query) or 1.0
            m_norms = np.linalg.norm(matrix, axis=1)
            m_norms[m_norms == 0] = 1.0
            return (matrix @ query) / (m_norms * q_norm)

        # euclidean: distance -> negate so higher = closer
        diffs = matrix - query
        distances = np.linalg.norm(diffs, axis=1)
        return -distances

    def __len__(self) -> int:
        return len(self._records)

    def name(self) -> str:
        return f"InMemory(metric={self.metric})"
