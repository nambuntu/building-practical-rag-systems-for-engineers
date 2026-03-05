# Chapter 9: Prompting for RAG (Prompt Contracts)

This lab compares `naive` and `contract` prompting for retrieval QA.

It measures:
- citation coverage
- refusal correctness on unanswerable queries
- injection success rate on an attack case
- strict format validity

## Setup

From inside `ch09_prompting_for_rag/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 run_lab.py
python3 run_lab.py --prompt naive
python3 run_lab.py --prompt contract
python3 run_lab.py --provider ollama --model llama3.2:3b
```

## Artifacts

Each run writes:

- `runs/<run_id>/config.json`
- `runs/<run_id>/results_naive.jsonl` (if mode executed)
- `runs/<run_id>/results_contract.jsonl` (if mode executed)
- `runs/<run_id>/report.json`
- `runs/<run_id>/run.log`

## What you should observe

- Contract mode usually has higher parseability (`format_ok_rate`).
- Contract mode should improve refusal correctness for unanswerable questions.
- Contract mode should lower injection success on the attack query.
- Citations become machine-checkable debugging signals.
