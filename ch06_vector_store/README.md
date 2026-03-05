# Chapter 06: Building Your First Vector Store (FAISS)

This lab builds retrieval manually: chunk text, embed chunks, index vectors in FAISS, run similarity search, and inspect ranked chunks with timings.

## Prerequisites

- Python 3.10+

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From inside `ch06_vector_store/`:

```bash
python3 run_lab.py
```

Example runs:

```bash
python3 run_lab.py --chunk-size 120 --overlap 20
python3 run_lab.py --top-k 3 --show-query Q3
python3 run_lab.py --no-compare-bruteforce
```

## What You Should Observe

- FAISS `IndexFlatIP` results match brute-force ranking on normalized vectors.
- Chunking parameters can change whether a query is answerable.
- For small corpora, embedding/index setup often dominates search time.
