# Feature 2: Embeddings + Vector Store

Continuation of Feature 1. The chunks are produced by **Feature 1's ingestion**
(load → clean → chunk); this feature turns those chunks into vectors, stores
them with metadata, and runs top-k similarity search.

## What this feature does

1. Reuse Feature 1 to get chunks (no re-chunking here)
2. Embed each chunk into a vector (2 strategies)
3. Store vectors + text + metadata in a vector store
4. Run top-k similarity search and compare keyword vs semantic retrieval

Run from the project root via the shared app:

```bash
python3 -m pip install -r requirements.txt

python3 main.py --feature 2 --embedder all --metric cosine
python3 main.py --feature 2 --chunker recursive --embedder semantic --no-metrics
```

## Pipeline

```
Feature 1 (reused)                Feature 2 (this feature)
load → clean → chunk    ───────►  embed → store → top-k search
```

Because everything lives in one project, this feature imports Feature 1's
ingestion directly: `from ..feature_1_document_ingestion.pipeline import
run_ingestion`. No cross-directory bridge is needed.

## Embedding strategies

- **Hashing (sparse / keyword)**: pure NumPy bag-of-words via the hashing trick.
  Fast, no download, but only matches on shared exact words.
- **SentenceTransformer (dense / semantic)**: real neural embeddings
  (`all-MiniLM-L6-v2`, 384 dims). Matches on meaning, so it handles synonyms
  and paraphrases.

## Vector store

- **InMemoryVectorStore**: brute-force, exact search over all vectors. Supports
  three similarity metrics: `cosine` (default), `dot`, `euclidean`. Each stored
  record keeps the vector, original text, and metadata (source, chunk index,
  char offsets).

Both embedders and stores follow the **Strategy Pattern** (same as Feature 1's
chunkers), so the pipeline never changes when you swap implementations.

## What the demo shows

- An embedder comparison table (dimension, chunk count, embed time)
- The same queries run through keyword vs semantic embeddings, side by side, so
  you can see dense embeddings win on paraphrased queries
- A metric comparison (cosine vs dot vs euclidean) on the same dense vectors

## Key ideas

- An embedding is just a fixed-length vector representing meaning.
- Similar meaning → nearby vectors, even with no shared words.
- Metadata stored beside each vector is what makes retrieval controllable.
- Brute-force search is exact; production uses approximate indexes (HNSW/IVF).
