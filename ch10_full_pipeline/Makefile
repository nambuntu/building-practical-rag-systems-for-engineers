.PHONY: demo-local demo-squad pgvector-up pgvector-down inspect-latest

demo-local:
	python3 start.py run --fast --source local --input-dir data/manuals --skip-eval --question "When should the ninety second timer be armed?"

demo-squad:
	python3 start.py run --fast --source squad --split train --sample-size 1500 --with-eval --eval-mode retrieval --question "what is so great about beyonce"

pgvector-up:
	docker compose up -d

pgvector-down:
	docker compose down

inspect-latest:
	python3 start.py inspect --run-id $$(ls -1 runs | sort | tail -n 1)
