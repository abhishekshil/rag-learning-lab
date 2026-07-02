import nltk
from sentence_transformers import SentenceTransformer, util

from .base_chunker import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 500, similarity_threshold: float = 0.15):
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")

        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def split(self, text: str) -> list[Chunk]:
        sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        if not sentences:
            return []

        embeddings = self._model.encode(sentences, convert_to_tensor=True)
        chunks: list[Chunk] = []
        current = [sentences[0]]
        search_from = 0

        for i in range(1, len(sentences)):
            similarity = float(util.cos_sim(embeddings[i - 1], embeddings[i]).item())
            candidate = " ".join(current + [sentences[i]])

            if len(candidate) > self.max_chunk_size or similarity < self.similarity_threshold:
                search_from = _append_chunk(chunks, text, " ".join(current), search_from)
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        _append_chunk(chunks, text, " ".join(current), search_from)
        return chunks

    def strategy_name(self) -> str:
        return f"Semantic(max={self.max_chunk_size}, sim>={self.similarity_threshold:.2f})"


def _append_chunk(chunks: list[Chunk], source: str, chunk_text: str, search_from: int) -> int:
    char_start = source.find(chunk_text[:40], search_from)
    if char_start == -1:
        char_start = search_from
    char_end = min(char_start + len(chunk_text), len(source))
    chunks.append(
        Chunk(
            text=chunk_text,
            index=len(chunks),
            char_start=char_start,
            char_end=char_end,
        )
    )
    return max(char_start + 1, char_end - 40)
