# Makes src/embedding a Python package.
# Import embedders here so callers can do:
#   from src.embedding import HashingEmbedder, SentenceTransformerEmbedder

from .base_embedder import BaseEmbedder
from .hashing_embedder import HashingEmbedder
from .sentence_transformer_embedder import SentenceTransformerEmbedder
