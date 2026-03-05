# Chapter 05: Chunking, The Most Underrated Problem

This lab shows how chunking choices change retrieval quality, including cases where a query becomes unanswerable because required context is split across chunk boundaries.

## Prerequisites

- Python 3.10+

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From inside `ch05_chunking/`:

```bash
python3 run_lab.py
```

Recommended runs:

```bash
python3 run_lab.py
python3 run_lab.py --chunker fixed --chunk-size 80 --overlap 0 --no-grid
python3 run_lab.py --chunker fixed --chunk-size 80 --overlap 40 --no-grid
python3 run_lab.py --chunker semantic --chunk-size 180 --no-grid
python3 run_lab.py --show-query Q3
```

## What You Should Observe

- Small fixed chunks with no overlap increase unanswerable queries.
- Overlap often improves recall by recovering boundary-split facts, but increases chunk count.
- Semantic chunking avoids sentence splits and can improve ranking quality.

## Why Synthetic Phrase Labels?

This chapter uses a synthetic manual and phrase-based gold labels so failures are measurable and deterministic: if required phrases do not exist in one chunk, that chunking config cannot fully answer the query. This mirrors real RAG failure modes where retrieval misses cross-boundary context even when embedding quality is acceptable.
