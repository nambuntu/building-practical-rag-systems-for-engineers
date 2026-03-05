# Build practical RAG systems for engineer code companion

This repo accompanies **Build practical RAG systems for engineer**.
It contains chapter-by-chapter labs you can run, break, and inspect without guesswork.

Repo: https://github.com/nambuntu/building-practical-rag-systems-for-engineers

## Chapters

| # | Title | Folder |
|---|---|---|
| 01 | Local model with Ollama | [`ch01_llm_basics/`](./ch01_llm_basics/) |
| 02 | Why LLM alone is not enough | [`ch02_llm_limitations/`](./ch02_llm_limitations/) |
| 03 | Embeddings without math pain | [`ch03_embeddings/`](./ch03_embeddings/) |
| 04 | Similarity search at scale | [`ch04_similarity_search/`](./ch04_similarity_search/) |
| 05 | Chunking tradeoffs | [`ch05_chunking/`](./ch05_chunking/) |
| 06 | First vector store (FAISS) | [`ch06_vector_store/`](./ch06_vector_store/) |
| 07 | Persistent vectors with pgvector | [`ch07_pgvector/`](./ch07_pgvector/) |
| 08 | Measuring retrieval quality | [`ch08_retrieval_quality/`](./ch08_retrieval_quality/) |
| 09 | Prompt contracts for RAG | [`ch09_prompting_for_rag/`](./ch09_prompting_for_rag/) |
| 10 | Full pipeline capstone | [`ch10_full_pipeline/`](./ch10_full_pipeline/) |

## Quickstart

```bash
git clone https://github.com/nambuntu/building-practical-rag-systems-for-engineers
cd building-practical-rag-systems-for-engineers
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell): use `.venv\\Scripts\\Activate.ps1` instead of `source`.

## Prerequisites

- Python 3.11+
- Optional: Ollama (for local LLM chapters)
- Optional: Docker + Docker Compose (for pgvector chapter and some pipeline setups)
- Optional: Make (helper targets in later chapters)

## How to run a chapter

```bash
cd chXX_<name>
python3 run_lab.py
```

Examples:

```bash
cd ch02_llm_limitations && python3 run_all.py
cd ch10_full_pipeline && python3 start.py run --source local --input-dir data/manuals --fast --skip-eval --question "When should the ninety second timer be armed?"
```

Outputs/logs are written in chapter-local folders (commonly `runs/` or chapter-specific output files).
