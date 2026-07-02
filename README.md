# RAG-Learning-Lab

Hands-on learning project for building Retrieval-Augmented Generation (RAG) systems step by step.

## What This Repository Is

This repo is a practical RAG learning path.
Each feature focuses on one core piece of a production RAG pipeline.

- `feature_1_document_ingestion`: load, clean, chunk, and compare chunking strategies

## Feature 1: Document Ingestion (Beginner View)

Goal: turn raw text documents into clean chunks that can be embedded and indexed later.

Pipeline:
1. Load document
2. Clean text
3. Split into chunks using different strategies
4. Compare chunk quality and chunk-size stats

Entry point:
- `feature_1_document_ingestion/run.py`

## Chunking Strategies Included (7)

1. **Fixed Size**
   - Splits by character count
   - Easiest baseline

2. **Sentence Based (NLTK)**
   - Keeps sentence boundaries
   - Good readability and low compute

3. **Sliding Window**
   - Fixed-size chunks with overlap
   - Helps keep boundary context

4. **Recursive Split**
   - Tries paragraph -> line -> space -> character
   - Practical default for mixed text

5. **Structure Aware (Markdown)**
   - Uses markdown headers/sections first
   - Best when docs are clearly structured

6. **Semantic Chunking (Embedding-based)**
   - Uses open-source model `all-MiniLM-L6-v2`
   - Splits when cosine similarity between neighboring sentences drops

7. **Parent-Document (Parent/Child)**
   - Small chunks for retrieval behavior, parent chunks for richer context
   - Useful for quality-focused RAG

## Quick Start

From repository root:

```bash
cd feature_1_document_ingestion
python3 -m pip install -r requirements.txt
python3 run.py
```

## What You Will See in Output

- Number of chunks per strategy
- Avg / min / max chunk size
- Full chunk previews for each strategy

This helps you understand how each strategy changes context boundaries.

## Strategy Selection Cheat Sheet

- Need simplest baseline -> **Fixed Size**
- Need sentence integrity -> **Sentence Based**
- Losing context at boundaries -> **Sliding Window**
- Want robust default for general prose -> **Recursive**
- Working with markdown headers/docs -> **Structure Aware**
- Need topic-shift awareness -> **Semantic**
- Need high precision + rich context -> **Parent-Document**

## Next Learning Step

After ingestion, the natural next step is:
- Embed chunks
- Store vectors in a vector DB
- Run retrieval + answer generation
