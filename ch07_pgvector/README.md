# Chapter 7: Persistent Storage with pgvector

This lab upgrades retrieval from in-memory FAISS to persistent Postgres + pgvector, then compares both backends on the same chunk/query set.

## Prerequisites

- Python 3.10+
- Docker + Docker Compose
- Make (recommended)

## Setup

From inside `ch07_pgvector/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make up
```

## Run

```bash
python3 run_lab.py
```

Example runs:

```bash
python3 run_lab.py --chunk-size 120 --overlap 20
python3 run_lab.py --reuse-db
python3 run_lab.py --repeat 200 --pg-index hnsw
```

## Make targets

- `make up`: start Postgres + pgvector
- `make down`: stop containers
- `make reset`: stop containers and delete DB volume
- `make logs`: tail Postgres logs
- `make psql`: open psql shell in container
- `make lab`: start DB and run lab script

## What You Should Observe

- pgvector keeps embeddings after restarts (persistent store).
- Top-k from pgvector and FAISS is identical or near-identical with normalized vectors.
- For small corpora, setup/embedding often costs more than search.
- ANN indexes trade build time for query speed as corpus size grows.
