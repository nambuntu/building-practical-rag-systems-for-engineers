# Chapter 04: Similarity Search at Scale

This lab compares brute-force cosine similarity against FAISS on synthetic vectors.

## Prerequisites

- Python 3.10+

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From inside `ch04_similarity_search/`:

```bash
python3 run_lab.py
```

Useful experiments:

```bash
python3 run_lab.py --n 5000
python3 run_lab.py --n 50000
python3 run_lab.py --index ivf --nprobe 2
python3 run_lab.py --index ivf --nprobe 32
```

## What to observe

- Brute-force query time grows roughly linearly with `N`.
- FAISS search stays much faster as corpus size grows.
- IVF can trade some Recall@K / MRR for speed, and `--nprobe` tunes that tradeoff.

## Notes

- Dataset cache is written to `data/ch04_dataset.npz`.
- Reuse cache is on by default; disable with `--no-reuse-dataset`.
