# Feature 2: Embeddings + Vector Store (continuation of Feature 1).
# Re-export the public pieces so callers (e.g. main.py) can do:
#   from src.feature_2_embeddings_vector_store import build_index, search

from .pipeline import build_index, chunks_to_records, search, IndexResult
from .embedding import HashingEmbedder, SentenceTransformerEmbedder
from .vector_store import InMemoryVectorStore, VectorRecord, SearchResult
