from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker, Chunk, docs_to_chunks


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split(self, text: str) -> list[Chunk]:
        return docs_to_chunks(self._splitter.create_documents([text]), text)

    def strategy_name(self) -> str:
        return f"Recursive(size={self.chunk_size}, overlap={self.chunk_overlap})"
