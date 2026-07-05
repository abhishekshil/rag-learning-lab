# Feature 2: Embeddings + Vector Store (continuation of Feature 1).
# Re-export the public pieces so callers (e.g. main.py) can do:
#   from src.feature_2_embeddings_vector_store import build_index, search

from .pipeline import build_index, chunks_to_records, search, IndexResult
from .embedding import (
    ALL_EMBEDDER_KEYS,
    CATEGORY_LABELS,
    EMBEDDER_CATEGORIES,
    ClipEmbedder,
    CodeEmbedder,
    HashingEmbedder,
    HybridEmbedder,
    InstructEmbedder,
    LangChainSentenceTransformerEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    SpladeEmbedder,
)
from .vector_store import InMemoryVectorStore, VectorRecord, SearchResult
