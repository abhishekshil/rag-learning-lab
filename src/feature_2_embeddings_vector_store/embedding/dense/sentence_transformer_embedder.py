# sentence_transformer_embedder.py
# Category: DENSE — direct sentence-transformers (no LangChain).
#
# Fast general-purpose semantic bi-encoder. Default: all-MiniLM-L6-v2 (384 dims).

import numpy as np

from ..base_embedder import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    category = "dense"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def name(self) -> str:
        short = self.model_name.split("/")[-1]
        return f"SentenceTransformer({short}, dim={self._dimension}, dense)"
