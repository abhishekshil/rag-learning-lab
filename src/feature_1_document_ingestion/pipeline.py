# pipeline.py
# Responsibility: Orchestrate the three steps of document ingestion:
#   Step 1 → Load the document (document_loader)
#   Step 2 → Clean the text    (text_cleaner)
#   Step 3 → Chunk the text    (any BaseChunker strategy)
#

from dataclasses import dataclass
from .document_loader import load_document
from .text_cleaner import clean_text
from .chunking.base_chunker import BaseChunker, Chunk


@dataclass
class IngestionResult:
    """
    Everything produced by one ingestion run.

    Attributes:
        file_path:      The source file.
        strategy_name:  Human-readable name of the chunker used.
        raw_char_count: Characters in the original text.
        clean_char_count: Characters after cleaning.
        chunks:         The list of Chunk objects produced.
    """
    file_path: str
    strategy_name: str
    raw_char_count: int
    clean_char_count: int
    chunks: list[Chunk]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def avg_chunk_size(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(len(c.text) for c in self.chunks) / len(self.chunks)

    @property
    def min_chunk_size(self) -> int:
        if not self.chunks:
            return 0
        return min(len(c.text) for c in self.chunks)

    @property
    def max_chunk_size(self) -> int:
        if not self.chunks:
            return 0
        return max(len(c.text) for c in self.chunks)


def run_ingestion(file_path: str, chunker: BaseChunker) -> IngestionResult:
    """
    Run the full ingestion pipeline on a single file.

    Args:
        file_path: Path to the document to ingest.
        chunker:   Any chunker that extends BaseChunker.

    Returns:
        An IngestionResult with all chunks and statistics.
    """
    # Step 1: Load
    raw_text = load_document(file_path)

    # Step 2: Clean
    clean = clean_text(raw_text)

    # Step 3: Chunk
    chunks = chunker.split(clean)

    return IngestionResult(
        file_path=file_path,
        strategy_name=chunker.strategy_name(),
        raw_char_count=len(raw_text),
        clean_char_count=len(clean),
        chunks=chunks,
    )
