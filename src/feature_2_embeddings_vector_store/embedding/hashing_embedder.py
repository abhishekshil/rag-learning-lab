# hashing_embedder.py
# A SPARSE / keyword-style baseline embedder. No model download, pure NumPy.
#
# How it works (the "hashing trick"):
#   1. Lowercase and split the text into word tokens.
#   2. Hash each token to a bucket index in [0, dimension).
#   3. Count how often each bucket is hit -> a term-frequency vector.
#   4. L2-normalize the vector so cosine similarity behaves well.
#
# This is essentially bag-of-words. It is fast and interpretable, but it can
# ONLY match on shared exact words. It has no idea that "car" and "automobile"
# mean the same thing. We include it so you can directly feel the gap between
# keyword matching and true semantic (dense) embeddings.

import hashlib
import re

import numpy as np

from .base_embedder import BaseEmbedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder(BaseEmbedder):
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
        # Stable hash across processes (Python's built-in hash() is salted).
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self._dimension

    def name(self) -> str:
        return f"Hashing(dim={self._dimension}, sparse/keyword)"
