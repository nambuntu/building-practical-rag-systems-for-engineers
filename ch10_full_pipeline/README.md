# Chapter 10: Full Pipeline Capstone

This chapter packages a complete stage-based RAG pipeline in one self-contained project.

Pipeline stages:
- ingest
- prepare
- index
- query
- evaluate (optional)

The key Chapter 10 upgrades are:
- prompt contracts (`naive` vs `contract`) in query stage
- machine-checkable query artifacts (`citations`, `refused`, `format_ok`)
- retrieval-only evaluation mode by default (`eval_mode=retrieval`)
- easy local-data demo path
- run artifact inspection command

## Setup

From inside `ch10_full_pipeline/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## One-command local demo

```bash
python3 start.py run \
  --source local \
  --input-dir data/manuals \
  --fast \
  --skip-eval \
  --question "When should the ninety second timer be armed?"
```

## One-command Hugging face squad demo

```bash
python3 start.py run \
  --fast --source squad \
  --split train \
  --sample-size 1500 \
  --with-eval \
  --eval-mode retrieval \
  --question "what is so great about beyonce"
```

## Prompt mode comparison

Run two queries against the same run id:

```bash
python3 start.py query --run-id <run_id> --prompt-mode naive --question "If coolant pump fails, which lever is used?"
python3 start.py query --run-id <run_id> --prompt-mode contract --question "If coolant pump fails, which lever is used?"
```

Then inspect:

```bash
cat runs/<run_id>/query/results.jsonl
```

## Evaluate modes

Retrieval-only (default for chapter quick runs):

```bash
python3 start.py evaluate --run-id <run_id> --eval-mode retrieval
```

Full mode (includes EM/F1 via LLM):

```bash
python3 start.py evaluate --run-id <run_id> --eval-mode full
```

## Inspect artifacts

```bash
python3 start.py inspect --run-id <run_id>
```

Expected run artifacts:

- `runs/<run_id>/manifest.json`
- `runs/<run_id>/ingest/raw.jsonl`
- `runs/<run_id>/prepare/chunks.jsonl`
- `runs/<run_id>/prepare/eval_examples.jsonl`
- `runs/<run_id>/index/index_manifest.json`
- `runs/<run_id>/query/results.jsonl`
- `runs/<run_id>/evaluate/report.json` (if evaluate ran)
- `runs/<run_id>/run.log`

## Make targets

```bash
make demo-local
make demo-squad
make pgvector-up
make pgvector-down
make inspect-latest
```
