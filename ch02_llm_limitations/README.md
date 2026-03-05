# Chapter 02: Why an LLM Alone Is Not Enough

This lab demonstrates why retrieval exists using only local components.

## Prerequisites

- Python 3.10+
- Ollama installed
- Local model pulled (default):

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
python3 run_all.py
python3 ask_baseline.py
python3 ask_manual_inject.py
python3 ask_manual_inject.py --include-filler
python3 ask_rag.py --top-k 3
python3 show_costs.py
```

## Config knobs

- `RAG_LLM_MODEL` (default: `llama3:8b-instruct-q4_0`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `CH2_TOP_K` (default: `3`)
- `CH2_PROMPT_TOKEN_LIMIT` (default: `3000`)

## What to observe

- Baseline should be uncertain or incorrect on private policy details.
- Manual injection can answer correctly but gets large quickly.
- Manual injection with filler should trigger context overflow.
- RAG should answer with much smaller prompt context.
