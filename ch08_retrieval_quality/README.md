# Chapter 8: Measuring Retrieval Quality

This lab evaluates retrieval quality with deterministic embeddings and FAISS. It focuses on retrieval metrics (Recall@k, MRR@k, naive Precision@k), not generation.

## Prerequisites

- Python 3.10+

## Setup

From inside `ch08_retrieval_quality/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 run_lab.py
```

## Experiment commands

```bash
python3 run_lab.py --gold-mode strict
python3 run_lab.py --gold-mode loose
python3 run_lab.py --chunk-size 90 --overlap 10 --ks 1 3 5 10
python3 run_lab.py --top-k 10 --show-top 5
```

## What to observe

- `strict` vs `loose` relevance mapping changes scores even when retrieval engine is unchanged.
- Larger `k` usually improves Recall@k but not necessarily MRR@k.
- Chunking settings can change which chunks count as relevant.
- `naive_Precision@k` is included for intuition, but retrieval is a candidate stage, so treat precision carefully.
