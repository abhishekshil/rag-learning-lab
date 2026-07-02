# base_chunker.py
# Responsibility: Define the contract (interface) that every chunker must follow.
#
# LLD Pattern: Strategy Pattern
#   - BaseChunker is the "strategy interface"
#   - FixedSizeChunker, SentenceChunker, RecursiveChunker, etc. are concrete strategies
#   - The pipeline only talks to BaseChunker, so swapping strategies costs zero changes
#
# Why ABC (Abstract Base Class)?
#   Python won't let you instantiate a class that inherits ABC and doesn't
#   implement all @abstractmethod methods. This catches mistakes at import time,
#   not at runtime in production.

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """
    A single piece of text cut from a document.

    Attributes:
        text:        The chunk content.
        index:       Position of this chunk in the document (0-based).
        char_start:  Character offset where this chunk begins in the original text.
        char_end:    Character offset where this chunk ends.
    """
    text: str
    index: int
    char_start: int
    char_end: int

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(index={self.index}, chars={self.char_start}-{self.char_end}, preview='{preview}...')"


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.

    Every chunker must implement `split()`.
    Constructor parameters (chunk_size, overlap, etc.) are up to each strategy.
    """

    @abstractmethod
    def split(self, text: str) -> list[Chunk]:
        """
        Split the given text into a list of Chunk objects.

        Args:
            text: The cleaned document text.

        Returns:
            A list of Chunk objects in document order.
        """
        pass

    def strategy_name(self) -> str:
        """Human-readable name of this strategy. Used in reports."""
        return self.__class__.__name__


def docs_to_chunks(docs: list[Any], source_text: str) -> list[Chunk]:
    """Convert LangChain documents to Chunk objects with safe offsets."""
    chunks: list[Chunk] = []
    search_from = 0

    for i, doc in enumerate(docs):
        text = doc.page_content
        if not text.strip():
            continue

        char_start = doc.metadata.get("start_index", -1)
        if char_start is None or char_start < 0:
            char_start = source_text.find(text[:40], search_from)
            if char_start == -1:
                char_start = search_from

        char_end = min(char_start + len(text), len(source_text))
        search_from = max(char_start + 1, char_end - 40)

        chunks.append(
            Chunk(
                text=text,
                index=i,
                char_start=char_start,
                char_end=char_end,
            )
        )

    return chunks
