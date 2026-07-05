# Feature 2: Embeddings + Vector Store

Continuation of Feature 1. The chunks are produced by **Feature 1's ingestion**
(load → clean → chunk); this feature turns those chunks into vectors, stores
them with metadata, and runs top-k similarity search.

## What this feature does

1. Reuse Feature 1 to get chunks (no re-chunking here)
2. Embed each chunk into a vector (9 strategies across 7 categories)
3. Store vectors + text + metadata in a vector store
4. Run top-k similarity search and compare retrieval approaches

Run from the project root via the shared app:

```bash
python3 -m pip install -r requirements.txt

python3 main.py --feature 2 --embedder all --metric cosine
python3 main.py --feature 2 --chunker recursive --embedder instruct --no-metrics
python3 main.py --feature 2 --embedder openai   # needs OPENAI_API_KEY
```

## Pipeline

```
Feature 1 (reused)                Feature 2 (this feature)
load → clean → chunk    ───────►  embed → store → top-k search
```

## Embedding catalog (by category)

| Category | CLI key | Class | Use case |
|----------|---------|-------|----------|
| **sparse** | `hashing` | `HashingEmbedder` | Dumb keyword baseline (NumPy, no download) |
| **sparse** | `splade` | `SpladeEmbedder` | Learned sparse — smarter keywords (SPLADE) |
| **dense** | `semantic` | `SentenceTransformerEmbedder` | Direct sentence-transformers (MiniLM, 384d) |
| **dense** | `langchain` | `LangChainSentenceTransformerEmbedder` | Same model via LangChain |
| **hybrid** | `hybrid` | `HybridEmbedder` | Hashing + semantic fused (896d) |
| **instruct** | `instruct` | `InstructEmbedder` | E5 query/passage prefixes (768d) |
| **multimodal** | `clip` | `ClipEmbedder` | CLIP text tower (512d) |
| **code** | `code` | `CodeEmbedder` | Code-tuned model via LangChain |
| **api** | `openai` | `OpenAIEmbedder` | OpenAI `text-embedding-3-small` |

### Folder layout

```
embedding/
├── sparse/       hashing, splade
├── dense/        semantic, langchain
├── hybrid/       hybrid
├── instruct/     instruct (E5 prefixes)
├── multimodal/   clip
├── code/         code
├── api/          openai
├── langchain_adapter.py   # LangChain → BaseEmbedder bridge
└── categories.py          # category taxonomy
```

LangChain is used wherever a partner package exists (`langchain-huggingface`,
`langchain-openai`). SPLADE uses `sentence-transformers` `SparseEncoder` directly
(no LangChain sparse class in partner packages yet).

## Vector store

- **InMemoryVectorStore**: brute-force, exact search over all vectors. Supports
  three similarity metrics: `cosine` (default), `dot`, `euclidean`.

Both embedders and stores follow the **Strategy Pattern** (same as Feature 1's
chunkers), so the pipeline never changes when you swap implementations.

## Key ideas

- An embedding is just a fixed-length vector representing meaning.
- **Same embedder** must index and query (instruct models use different prefixes).
- Different dimensions are fine — each index is self-contained.
- Brute-force search is exact; production uses approximate indexes (HNSW/IVF).
