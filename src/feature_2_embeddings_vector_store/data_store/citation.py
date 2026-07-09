# citation.py
# A "citation" isn't stored separately — it's rendered from the metadata that every
# VectorRecord already carries (source file, chunk index, char offsets). Keeping this
# in one helper means every store backend produces identical, traceable source references.

from __future__ import annotations

from .base_store import SearchResult


def format_citation(result: SearchResult) -> str:
    """Render a single search hit as a compact, traceable source reference.

    Example: `sample.txt · chunk 3 · chars 1240–1690`
    """
    meta = result.record.metadata or {}
    source = meta.get("source", "unknown")
    parts = [str(source)]

    chunk_index = meta.get("chunk_index")
    if chunk_index is not None:
        parts.append(f"chunk {chunk_index}")

    char_start = meta.get("char_start")
    char_end = meta.get("char_end")
    if char_start is not None and char_end is not None:
        parts.append(f"chars {char_start}\u2013{char_end}")

    doc_id = meta.get("doc_id")
    if doc_id:
        parts.append(str(doc_id))

    return " \u00b7 ".join(parts)
