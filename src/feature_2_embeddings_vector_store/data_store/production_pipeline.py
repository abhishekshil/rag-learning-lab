# production_pipeline.py
# Tutorial: pgvector + Neo4j + MinIO over a research paper.
# Requires: docker compose up -d

from __future__ import annotations

from dataclasses import dataclass

from ...feature_1_document_ingestion.pipeline import run_ingestion
from ..embedding.base_embedder import BaseEmbedder
from ..embedding.langchain_adapter import embedder_as_langchain
from ..pipeline import IndexResult, chunks_to_records, search
from .base_store import BaseVectorStore, SearchResult
from .config import PAPER_DOC_ID, PAPER_PATH, PAPER_SOURCE_URI
from .minio_blob import MinioBlobStore
from .neo4j_graph import GraphHit, Neo4jCitationGraph
from .pgvector_store import PgVectorStore


@dataclass
class ProductionIndexResult:
    """Everything created by the production tutorial ingest."""
    index: IndexResult
    media_uri: str
    graph_edges: int
    blob_store_name: str
    graph_store_name: str


def make_production_store(
    embedder: BaseEmbedder,
    *,
    metric: str = "cosine",
    reset: bool = True,
) -> PgVectorStore:
    safe_name = embedder.name().replace("(", "_").replace(")", "").replace(",", "")
    return PgVectorStore(
        metric=metric,
        collection_name=f"tutorial_{safe_name}",
        embeddings=embedder_as_langchain(embedder),
        reset=reset,
    )


def build_production_index(
    file_path: str,
    chunker,
    embedder: BaseEmbedder,
    *,
    metric: str = "cosine",
    reset: bool = True,
) -> ProductionIndexResult:
    blob = MinioBlobStore()
    media_uri = blob.upload_file(file_path)

    graph = Neo4jCitationGraph(reset=reset)
    graph_edges = graph.seed_tutorial_graph()

    store = make_production_store(embedder, metric=metric, reset=reset)
    ingestion = run_ingestion(file_path, chunker)

    records, embed_seconds = chunks_to_records(
        ingestion.chunks,
        embedder,
        file_path,
        doc_id=PAPER_DOC_ID,
        source_uri=PAPER_SOURCE_URI,
        media_uri=media_uri,
        visibility="public",
    )
    store.add(records)

    return ProductionIndexResult(
        index=IndexResult(
            embedder_name=embedder.name(),
            store=store,
            chunk_count=len(records),
            dimension=embedder.dimension,
            embed_seconds=embed_seconds,
        ),
        media_uri=media_uri,
        graph_edges=graph_edges,
        blob_store_name=blob.name(),
        graph_store_name=graph.name(),
    )


def search_with_graph_expansion(
    store: BaseVectorStore,
    graph: Neo4jCitationGraph,
    embedder: BaseEmbedder,
    query: str,
    *,
    k: int = 3,
    graph_seed: str = "FlashAttention",
) -> tuple[list[SearchResult], list[GraphHit]]:
    vector_hits = search(store, embedder, query, k=k)
    graph_hits = graph.expand(graph_seed, hops=1)
    return vector_hits, graph_hits


def fetch_raw_paper(blob: MinioBlobStore, media_uri: str) -> str:
    return blob.download_text(media_uri)


DEFAULT_PAPER = PAPER_PATH
