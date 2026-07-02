import nltk
from langchain_text_splitters import NLTKTextSplitter
from .base_chunker import BaseChunker, Chunk, docs_to_chunks

# Needed by NLTK sentence tokenization.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class SentenceChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 500, chunk_overlap: int = 50):
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = NLTKTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def split(self, text: str) -> list[Chunk]:
        return docs_to_chunks(self._splitter.create_documents([text]), text)

    def strategy_name(self) -> str:
        return f"Sentence/NLTK(max={self.max_chunk_size}, overlap={self.chunk_overlap})"
