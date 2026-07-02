# Feature 1: Document Ingestion

Beginner-friendly implementation of document loading, cleaning, and chunking strategy comparison.

## Goal

Turn raw document text into clean, retrieval-ready chunks and understand how chunking strategy changes output quality.

Pipeline:
1. Load document
2. Clean text
3. Split into chunks using different strategies
4. Compare chunk stats and chunk boundaries

Entry point:
- `run.py`

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

From this feature folder:

```bash
python3 -m pip install -r requirements.txt
python3 run.py
```

## What You Will See in Output

- Number of chunks per strategy
- Avg / min / max chunk size
- Full chunk previews for each strategy

This gives an immediate understanding of how each strategy affects retrieval context.

## Strategy Selection Cheat Sheet

- Need simplest baseline -> **Fixed Size**
- Need sentence integrity -> **Sentence Based**
- Losing context at boundaries -> **Sliding Window**
- Want robust default for general prose -> **Recursive**
- Working with markdown headers/docs -> **Structure Aware**
- Need topic-shift awareness -> **Semantic**
- Need high precision + rich context -> **Parent-Document**
