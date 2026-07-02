from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker, Chunk


class StructureAwareChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False,
        )
        self._size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split(self, text: str) -> list[Chunk]:
        header_docs = self._header_splitter.split_text(text)
        chunks: list[Chunk] = []
        search_from = 0

        for doc in header_docs:
            section_text = doc.page_content
            if not section_text.strip():
                continue
            for section_doc in self._size_splitter.create_documents([section_text]):
                chunk_text = section_doc.page_content
                if not chunk_text.strip():
                    continue
                char_start = text.find(chunk_text[:40], search_from)
                if char_start == -1:
                    char_start = search_from
                char_end = min(char_start + len(chunk_text), len(text))
                search_from = max(char_start + 1, char_end - 40)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=len(chunks),
                        char_start=char_start,
                        char_end=char_end,
                    )
                )

        return chunks

    def strategy_name(self) -> str:
        return f"StructureAware/Markdown(size={self.chunk_size}, overlap={self.chunk_overlap})"
