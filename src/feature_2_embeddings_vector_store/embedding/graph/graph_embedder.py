# graph_embedder.py
# Category: GRAPH — LangChain text embedding + citation-graph structure fused.
#
# Text similarity alone cannot answer "what improved on FlashAttention?" — the
# graph side encodes which papers cite / improve on which. We concat a Node2Vec
# structural vector over the tutorial citation graph with a LangChain
# HuggingFaceEmbeddings text vector.

from __future__ import annotations

import numpy as np

from ...data_store.citation_data import TUTORIAL_CITATIONS, TUTORIAL_CHUNK_LINKS
from ..langchain_adapter import LangChainEmbedder


class GraphEmbedder(LangChainEmbedder):
    """Fuse LangChain dense text vectors with Node2Vec citation-graph embeddings."""

    category = "graph"
    GRAPH_DIM = 64

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        short = model_name.split("/")[-1]
        super().__init__(
            embeddings,
            display_name=f"Graph+LangChain({short})",
            category="graph",
        )
        self.model_name = model_name
        self._paper_vectors = self._compute_node2vec(self.GRAPH_DIM)
        self._chunk_to_papers = self._invert_chunk_links()
        self._dimension = super().dimension + self.GRAPH_DIM

    @staticmethod
    def _invert_chunk_links() -> dict[str, list[str]]:
        """Map chunk id → paper names that reference it."""
        out: dict[str, list[str]] = {}
        for paper, chunk_ids in TUTORIAL_CHUNK_LINKS.items():
            for cid in chunk_ids:
                out.setdefault(cid, []).append(paper)
        return out

    @staticmethod
    def _build_citation_graph():
        """Directed citation graph: papers as nodes, citation edges as arcs."""
        import networkx as nx

        G = nx.DiGraph()
        for paper in TUTORIAL_CHUNK_LINKS:
            G.add_node(paper)
        for src, dst, rel in TUTORIAL_CITATIONS:
            G.add_edge(src, dst, relation=rel)
        return G

    @classmethod
    def _compute_node2vec(cls, dim: int = 64) -> dict[str, np.ndarray]:
        """Node2Vec over the tutorial citation graph (random-walk + skip-gram)."""
        from node2vec import Node2Vec

        G = cls._build_citation_graph()
        n2v = Node2Vec(
            G,
            dimensions=dim,
            walk_length=max(3, min(10, G.number_of_nodes() * 2)),
            num_walks=50,
            p=1.0,
            q=1.0,
            workers=1,
            seed=42,
            quiet=True,
        )
        model = n2v.fit(window=3, min_count=1, batch_words=4, seed=42)

        vectors: dict[str, np.ndarray] = {}
        for paper in TUTORIAL_CHUNK_LINKS:
            vec = model.wv[paper].astype(np.float32)
            norm = np.linalg.norm(vec) or 1.0
            vectors[paper] = vec / norm
        return vectors

    @staticmethod
    def _papers_mentioned_in_text(text: str) -> list[str]:
        lower = text.lower()
        return [p for p in TUTORIAL_CHUNK_LINKS if p.lower() in lower]

    def _graph_vector_for_chunk(self, chunk_id: str, text: str) -> np.ndarray:
        linked = set(self._chunk_to_papers.get(chunk_id, []))
        linked.update(self._papers_mentioned_in_text(text))
        if not linked:
            return np.zeros(self.GRAPH_DIM, dtype=np.float32)
        vec = np.mean([self._paper_vectors[p] for p in linked], axis=0)
        norm = np.linalg.norm(vec) or 1.0
        return (vec / norm).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _fuse(self, text_vectors: np.ndarray, graph_vectors: np.ndarray) -> np.ndarray:
        return np.hstack([text_vectors, graph_vectors]).astype(np.float32)

    def embed(self, texts: list[str], *, chunk_ids: list[str] | None = None) -> np.ndarray:
        text_vecs = super().embed(texts)
        graph_vecs = []
        for i, text in enumerate(texts):
            cid = (chunk_ids or [None] * len(texts))[i] or f"chunk-{i}"
            graph_vecs.append(self._graph_vector_for_chunk(cid, text))
        return self._fuse(text_vecs, np.vstack(graph_vecs))

    def embed_one(self, text: str) -> np.ndarray:
        text_vec = super().embed_one(text).reshape(1, -1)
        mentioned = self._papers_mentioned_in_text(text)
        if mentioned:
            gv = np.mean([self._paper_vectors[p] for p in mentioned], axis=0)
            norm = np.linalg.norm(gv) or 1.0
            graph_vec = (gv / norm).reshape(1, -1).astype(np.float32)
        else:
            graph_vec = np.zeros((1, self.GRAPH_DIM), dtype=np.float32)
        return self._fuse(text_vec, graph_vec)[0]
