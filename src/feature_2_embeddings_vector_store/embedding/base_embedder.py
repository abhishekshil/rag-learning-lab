# base_embedder.py
# Responsibility: Define the contract every embedding model must follow.
#
# LLD Pattern: Strategy Pattern (same idea as Feature 1's chunkers)
#   - BaseEmbedder is the "strategy interface"
#   - HashingEmbedder, SentenceTransformerEmbedder, HybridEmbedder are strategies
#   - The pipeline and vector store only talk to BaseEmbedder, so swapping a
#     keyword baseline for a real neural model costs zero changes elsewhere.
#
# An "embedding" is just a fixed-length vector of floats that represents the
# meaning of a piece of text. Every embedder here returns a 2D NumPy array of
# shape (n_texts, dimension), where each row is one text's vector.

from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for all embedding strategies."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Convert a list of texts into a matrix of vectors.

        Args:
            texts: The texts to embed (chunks, or a single query in a list).

        Returns:
            A float32 array of shape (len(texts), dimension).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of each embedding vector."""
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience: embed a single text and return a 1D vector."""
        return self.embed([text])[0]

    def name(self) -> str:
        """Human-readable name of this embedder. Used in reports."""
        return self.__class__.__name__
