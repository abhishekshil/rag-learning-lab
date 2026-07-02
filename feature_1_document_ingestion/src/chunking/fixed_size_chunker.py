from langchain_text_splitters import CharacterTextSplitter
from .base_chunker import BaseChunker, Chunk, docs_to_chunks


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

        self._splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True,
        )

    def split(self, text: str) -> list[Chunk]:
        return docs_to_chunks(self._splitter.create_documents([text]), text)

    def strategy_name(self) -> str:
        return f"FixedSize(size={self.chunk_size})"
