# citation_data.py
# Shared tutorial citation graph — used by Neo4j seeding and graph embedder.

TUTORIAL_CITATIONS = [
    ("FlashAttention-2", "FlashAttention", "improves_on"),
    ("FlashAttention", "Transformer", "builds_on"),
    ("FlashAttention-2", "Transformer", "builds_on"),
]

TUTORIAL_CHUNK_LINKS = {
    "FlashAttention": ["chunk-2", "chunk-3", "chunk-4", "chunk-5"],
    "FlashAttention-2": ["chunk-2"],
    "Transformer": ["chunk-1", "chunk-2"],
}
