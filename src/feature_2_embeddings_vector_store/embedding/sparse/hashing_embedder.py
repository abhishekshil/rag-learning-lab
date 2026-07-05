# hashing_embedder.py
# Category: SPARSE — dumb keyword baseline (bag-of-words via hashing trick).
#
# No model download, pure NumPy. Only matches on shared exact words.

import hashlib
import re

import numpy as np

from ..base_embedder import BaseEmbedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder(BaseEmbedder):
    category = "sparse"

    def __init__(self, dimension: int = 512):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)

        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                bucket = self._hash(token)
                vectors[row, bucket] += 1.0

            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm

        return vectors

    def _hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self._dimension

    def name(self) -> str:
        return f"Hashing(dim={self._dimension}, sparse/keyword)"
