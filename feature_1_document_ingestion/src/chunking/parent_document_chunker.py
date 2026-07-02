from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker, Chunk, docs_to_chunks


class ParentDocumentChunker(BaseChunker):
    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 250,
        child_chunk_overlap: int = 40,
    ):
        if parent_chunk_size <= 0 or child_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        if parent_chunk_overlap < 0 or child_chunk_overlap < 0:
            raise ValueError("chunk overlaps cannot be negative")
        if parent_chunk_overlap >= parent_chunk_size:
            raise ValueError("parent_chunk_overlap must be smaller than parent_chunk_size")
        if child_chunk_overlap >= child_chunk_size:
            raise ValueError("child_chunk_overlap must be smaller than child_chunk_size")
        if child_chunk_size >= parent_chunk_size:
            raise ValueError("child_chunk_size must be smaller than parent_chunk_size")

        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split(self, text: str) -> list[Chunk]:
        parent_docs = self._parent_splitter.create_documents([text])
        for parent_doc in parent_docs:
            self._child_splitter.create_documents([parent_doc.page_content])
        return docs_to_chunks(parent_docs, text)

    def strategy_name(self) -> str:
        return (
            f"ParentDoc(parent={self.parent_chunk_size}/{self.parent_chunk_overlap}, "
            f"child={self.child_chunk_size}/{self.child_chunk_overlap})"
        )
