# Production-style vector store: Postgres + pgvector via LangChain.
# Embedder produces vectors; store persists and searches them via add_embeddings.

from __future__ import annotations

from typing import Any

from .base_store import BaseVectorStore, VectorRecord, SearchResult
from ..embedding.langchain_adapter import default_langchain_embeddings
from .config import POSTGRES_CONNECTION


class PgVectorStore(BaseVectorStore):
    """Persistent vector store on Postgres + pgvector (LangChain PGVector)."""

    def __init__(
        self,
        metric: str = "cosine",
        *,
        collection_name: str = "rag_lab",
        connection: str | None = None,
        embeddings: Any = None,
        reset: bool = False,
    ):
        if metric not in ("cosine", "dot", "euclidean"):
            raise ValueError(f"metric must be cosine|dot|euclidean, got '{metric}'")

        from langchain_postgres import PGVector

        self.metric = metric
        self.collection_name = collection_name
        self._connection = connection or POSTGRES_CONNECTION
        self._embeddings = embeddings or default_langchain_embeddings()
        self._store = PGVector(
            embeddings=self._embeddings,
            collection_name=collection_name,
            connection=self._connection,
            use_jsonb=True,
            pre_delete_collection=reset,
        )
        self._count = 0

    def add(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._store.add_embeddings(
            texts=[r.text for r in records],
            embeddings=[r.vector.tolist() for r in records],
            metadatas=[dict(r.metadata) for r in records],
            ids=[r.id for r in records],
        )
        self._count += len(records)

    def search(self, query_vector, k: int = 3) -> list[SearchResult]:
        if len(self) == 0:
            return []

        docs_and_scores = self._store.similarity_search_with_score_by_vector(
            embedding=list(map(float, query_vector)),
            k=min(k, len(self)),
        )
        hits: list[SearchResult] = []
        for doc, distance in docs_and_scores:
            meta = dict(doc.metadata or {})
            record = VectorRecord(
                id=str(doc.id or meta.get("chunk_index", "unknown")),
                vector=query_vector,
                text=doc.page_content,
                metadata=meta,
            )
            score = -float(distance) if self.metric == "euclidean" else 1.0 - float(distance)
            hits.append(SearchResult(record=record, score=score))
        return hits

    def __len__(self) -> int:
        return self._count

    def name(self) -> str:
        return f"PgVector(metric={self.metric}, collection={self.collection_name})"
