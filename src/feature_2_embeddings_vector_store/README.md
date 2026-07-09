# Feature 2: Embeddings + Vector Store

Continuation of Feature 1. The chunks are produced by **Feature 1's ingestion**
(load → clean → chunk); this feature turns those chunks into vectors, stores
them with metadata, and runs top-k similarity search.

This feature also includes a **production storage tutorial** based on
[production-rag-storage-choices.md](../../docs/production-rag-storage-choices.md)
— vector search alone is not enough; you also need blob pointers and a citation graph.

## What this feature does

1. Reuse Feature 1 to get chunks (no re-chunking here)
2. Embed each chunk into a vector (9 strategies across 7 categories)
3. Store vectors + text + metadata in a vector store
4. Run top-k similarity search and compare retrieval approaches
5. **(Tutorial)** Index a research paper across Postgres+pgvector, Neo4j, and MinIO

Run from the project root via the shared app:

```bash
python3 -m pip install -r requirements.txt

docker compose up -d

# Postgres + pgvector (LangChain) — default store
python3 main.py --feature 2 --embedder langchain --store pgvector

# Graph embedder: LangChain text + citation structure
python3 main.py --feature 2 --embedder graph --store production
```

## Production storage tutorial

Inspired by building RAG over AI research papers (text + citations + raw files).

```
Query → Orchestrator
          ├── pgvector:  embeddings + text + metadata (doc_id, content_hash, visibility, …)
          ├── Neo4j:     citation graph (FlashAttention-2 → improves_on → FlashAttention)
          └── MinIO:     raw paper file (chunks store s3:// pointers, not the file)
```

| Layer | Technology | LangChain package |
|-------|------------|-------------------|
| Vector | Postgres + pgvector | `langchain-postgres` (`PGVector`) |
| Graph | Neo4j | `langchain-neo4j` (`Neo4jGraph`) |
| Blob | MinIO (S3 API) | `boto3` (S3-compatible) |

**Start Docker:**

```bash
docker compose up -d
```

Services: Postgres `:5432`, Neo4j `:7687` / UI `:7474`, MinIO `:9000` / console `:9001`.
Copy `.env.example` to `.env` if you need custom credentials.

Tutorial paper: `docs/papers/flashattention-tutorial.txt` (arXiv:2205.14135 excerpt).

## Pipeline

```
Feature 1 (reused)                Feature 2 (this feature)
load → clean → chunk    ───────►  embed → store → top-k search
                                         └── (tutorial) blob + graph
```

## Embedding catalog (by category)

| Category | CLI key | Class | Use case |
|----------|---------|-------|----------|
| **sparse** | `hashing` | `HashingEmbedder` | Dumb keyword baseline (NumPy, no download) |
| **sparse** | `splade` | `SpladeEmbedder` | Learned sparse — smarter keywords (SPLADE) |
| **dense** | `langchain` | `LangChainSentenceTransformerEmbedder` | Same model via LangChain (default) |
| **dense** | `semantic` | `SentenceTransformerEmbedder` | Direct sentence-transformers (MiniLM, 384d) |
| **graph** | `graph` | `GraphEmbedder` | LangChain text + citation-graph structure (448d) |
| **hybrid** | `hybrid` | `HybridEmbedder` | Hashing + semantic fused (896d) |
| **instruct** | `instruct` | `InstructEmbedder` | E5 query/passage prefixes (768d) |
| **multimodal** | `clip` | `ClipEmbedder` | CLIP text tower (512d) |
| **code** | `code` | `CodeEmbedder` | Code-tuned model via LangChain |
| **api** | `openai` | `OpenAIEmbedder` | OpenAI `text-embedding-3-small` |

### Folder layout

```
embedding/           # 10 embedder strategies (langchain default, graph fuses citations)
data_store/        # all storage: pgvector, neo4j, minio, production_pipeline
```

## Vector store

Select the backend with `--store` (CLI) or the **Vector store** dropdown (UI).

| CLI key | Class | What it is |
|---------|-------|------------|
| `pgvector` (default) | `PgVectorStore` | Postgres + pgvector via LangChain `PGVector`. |
| `production` | `build_production_index()` | pgvector + Neo4j + MinIO over the tutorial paper. |

Both stores require `docker compose up -d`. They support three similarity metrics: `cosine` (default), `dot`, `euclidean`.

Stores receive **precomputed vectors** from the embedder — LangChain persists them via
`add_embeddings`, keeping embedder and store independent.

### Production chunk metadata

Every chunk stored in pgvector carries the fields from the blog:

| Field | Purpose |
|-------|---------|
| `text` | LLM reads text, not vectors |
| `doc_id` | Delete/update a whole paper at once |
| `source_uri` | arXiv link for citations |
| `content_hash` | Skip re-embedding unchanged chunks |
| `embed_model` | Never mix incompatible vector spaces |
| `visibility` | ACL filter at search time |
| `media_uri` | `s3://` pointer to raw file in MinIO |

### Source & citations

`format_citation(SearchResult)` renders metadata for display:

```
flashattention-tutorial.txt · chunk 3 · chars 1240–1690 · arxiv2205.14135
```

## Key ideas

- An embedding is just a fixed-length vector representing meaning.
- **Same embedder** must index and query (instruct models use different prefixes).
- Vectors alone are useless without text + metadata + blob pointers + graph edges.
- Citation questions need a graph layer — similarity search cannot answer "what improved on X?"
