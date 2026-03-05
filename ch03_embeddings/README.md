# Chapter 03: Embeddings Without Math Pain

A tiny lab that turns text into vectors, computes cosine similarity, and shows that similar meaning sits closer in vector space.

## Prerequisites

- Python 3.10+
- Ollama installed and running locally
- Pull the model:

```bash
ollama pull llama3:8b-instruct-q4_0
```

Start Ollama:

```bash
ollama serve
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 run_lab.py
```

Optional flags:

```bash
python3 run_lab.py --model llama3:8b-instruct-q4_0 --base-url http://localhost:11434 --timeout 120
```

## What to expect

- Embedding vector length for each sentence
- Preview of first 5 embedding values
- Pairwise cosine similarity matrix
- Interpretation with highest and lowest similarity pair

## Modify sentences

Edit the `SENTENCES` list in `run_lab.py` and rerun.
