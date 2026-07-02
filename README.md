# RAG Learning Lab

Learn RAG by implementation, feature by feature.

## Purpose

This project is a practical, build-first journey to understand how modern RAG systems are designed, implemented, debugged, and evaluated.

## Learning Path (Feature-wise)

Status legend:
- `[x]` done
- `[ ]` planned

### Foundation: Retrieval Pipeline

- [x] **Feature 1 - Document Ingestion + Chunking**
  - Load and clean raw documents
  - Compare 7 chunking strategies
  - Understand chunk boundaries and retrieval impact
  - Path: `feature_1_document_ingestion`
  - README: `feature_1_document_ingestion/README.md`

- [ ] **Feature 2 - Embeddings + Vector Store**
  - Generate embeddings for chunks
  - Store vectors with metadata
  - Run top-k similarity search

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
