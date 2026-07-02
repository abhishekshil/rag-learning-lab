# langchain_sentence_transformer_embedder.py
# DENSE / semantic embedder via LangChain's HuggingFaceEmbeddings wrapper.
#
# Same model and engine as SentenceTransformerEmbedder (all-MiniLM-L6-v2), but
# goes through LangChain's Embeddings interface (embed_documents / embed_query)
# instead of calling sentence-transformers directly.
#
# Compare side by side:
#   - SentenceTransformerEmbedder  → direct .encode() → np.ndarray
#   - LangChainSentenceTransformerEmbedder → LangChain → same model → np.ndarray

import numpy as np

from ..base_embedder import BaseEmbedder


class LangChainSentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Lazy import so hashing-only runs still work without langchain installed.
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._dimension = len(self._embeddings.embed_query("probe"))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._embeddings.embed_documents(texts)
        return np.array(vectors, dtype=np.float32)

    def name(self) -> str:
        short = self.model_name.split("/")[-1]
        return f"LangChainST({short}, dim={self._dimension}, dense/semantic)"
