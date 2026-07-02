# RAG Learning Lab

Learn RAG by implementation, feature by feature.

## Purpose

This project is a practical, build-first journey to understand how modern RAG systems are designed, implemented, debugged, and evaluated.

## Project layout

It is a single project. Common files live at the root; each feature is a
subpackage under `src/feature_*` containing only feature-specific code.

```
rag-learning-lab/
├── main.py            # the app: pick a feature + options, runs it
├── requirements.txt   # common dependencies (all features)
├── docs/              # RAG source documents, shared across features
│   └── sample.txt
└── src/
    ├── feature_1_document_ingestion/       # load, clean, 7 chunking strategies
    └── feature_2_embeddings_vector_store/  # embeddings, vector store, search
```

## How to run

```bash
python3 -m pip install -r requirements.txt

# Interactive menu (choose feature + options):
python3 main.py

# Or pass options directly:
python3 main.py --feature 1 --chunker all
python3 main.py --feature 1 --chunker recursive --show-chunks
python3 main.py --feature 2 --embedder all --metric cosine
python3 main.py --feature 2 --chunker recursive --embedder semantic --no-metrics
```

## Learning Path (Feature-wise)

Status legend:
- `[x]` done
- `[ ]` planned

### Foundation: Retrieval Pipeline

- [x] **Feature 1 - Document Ingestion + Chunking**
  - Load and clean raw documents
  - Compare 7 chunking strategies
  - Understand chunk boundaries and retrieval impact
  - Path: `src/feature_1_document_ingestion`
  - README: `src/feature_1_document_ingestion/README.md`
  - Run: `python3 main.py --feature 1 --chunker all`

- [x] **Feature 2 - Embeddings + Vector Store**
  - Generate embeddings for chunks (keyword vs semantic)
  - Store vectors with metadata
  - Run top-k similarity search
  - Reuses Feature 1's chunks (continuation)
  - Path: `src/feature_2_embeddings_vector_store`
  - README: `src/feature_2_embeddings_vector_store/README.md`
  - Run: `python3 main.py --feature 2 --embedder all`

- [ ] **Feature 3 - Retrieval + Grounded Answering**
  - Build query pipeline
  - Retrieve relevant chunks
  - Add prompt context and answer generation

### Agent System Design (Harness + Loop + Memory)

- [ ] **Feature 4 - Agent Harness**
  - System prompt, user prompt, chat history
  - Working memory / context window management

- [ ] **Feature 5 - Tool Loop**
  - Tool call -> result -> decide next step loop
  - Basic loop guardrails and stop conditions

- [ ] **Feature 6 - Memory Layers**
  - Procedural memory (rules/instructions)
  - Semantic memory (user/profile facts)
  - Episodic memory (past events and task context)

### LLM Ops (Trace -> Eval -> Diagnose -> Improve)

- [ ] **Feature 7 - Tracing + Observability**
  - Capture traces and token/latency metrics
  - Monitor quality and cost

- [ ] **Feature 8 - Evaluation Pipeline**
  - Retrieval and answer quality checks
  - Regression-style evals before changes

- [ ] **Feature 9 - Improvement + Release Loop**
  - Diagnose failures
  - Update prompts/retrieval config
  - Re-evaluate and release
