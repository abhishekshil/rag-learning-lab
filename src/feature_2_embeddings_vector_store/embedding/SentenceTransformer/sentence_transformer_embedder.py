# sentence_transformer_embedder.py
# DENSE / semantic embedder WITHOUT LangChain — direct sentence-transformers.
#
# It wraps a sentence-transformers model (default: all-MiniLM-L6-v2, 384 dims).
# For the LangChain wrapper over the same model, see
# langchain_sentence_transformer_embedder.py.
# The model was trained so that sentences with similar MEANING land close
# together in vector space, even when they share no words. This is what lets a
# query about "storing vectors with metadata" match a chunk about vector stores
# without the exact phrase appearing.
#
# The model is downloaded once (~90MB) and cached locally by huggingface.

import numpy as np

from ..base_embedder import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Imported lazily so the keyword baseline can run without torch installed.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True returns unit-length vectors, so cosine
        # similarity and dot product agree and scores land in a clean range.
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def name(self) -> str:
        short = self.model_name.split("/")[-1]
        return f"SentenceTransformer({short}, dim={self._dimension}, dense/semantic)"
