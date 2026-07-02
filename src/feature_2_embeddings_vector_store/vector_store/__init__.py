# Makes src/vector_store a Python package.
# Import here so callers can do:
#   from src.vector_store import InMemoryVectorStore, VectorRecord, SearchResult

from .base_store import BaseVectorStore, VectorRecord, SearchResult
from .in_memory_store import InMemoryVectorStore
