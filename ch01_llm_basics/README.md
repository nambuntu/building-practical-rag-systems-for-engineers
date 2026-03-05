# Chapter 01: Local AI Lab (Ollama + Python)

This chapter is a standalone local AI lab that shows how to call an Ollama model from Python.

## Prerequisites

- Python 3.10+
- Ollama installed and running locally
- A local model pulled, for example:

```bash
ollama pull llama3:latest
```

Start Ollama if needed:

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
python3 client.py
python3 stream_client.py
python3 promptbench.py
```

## Configuration

All scripts support CLI flags and environment overrides.

- Environment:
  - `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- Common flags:
  - `--model` (default: `llama3:latest`)
  - `--base-url`
  - `--timeout` (default: `120`)
- `promptbench.py` also supports:
  - `--output` (default: `results/results.csv`)

Examples:

```bash
python3 client.py --model llama3:latest --prompt "Answer with exactly: hello"
python3 stream_client.py --timeout 180
python3 promptbench.py --output results/results.csv
```

## Expected Outputs

- `client.py`: prints one JSON object with model, response, duration, and optional token counts.
- `stream_client.py`: streams text to terminal, then prints total duration and optional counts.
- `promptbench.py`: writes CSV benchmark rows to `results/results.csv`.

## Troubleshooting

If you see connection errors to `localhost:11434`:

1. Ensure Ollama is running (`ollama serve`).
2. Ensure the model exists (`ollama list`).
3. Confirm your URL (`OLLAMA_BASE_URL` or `--base-url`).
