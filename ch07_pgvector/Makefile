COMPOSE=docker compose
SERVICE=pgvector

.PHONY: up down reset logs psql lab

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

reset:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f $(SERVICE)

psql:
	$(COMPOSE) exec $(SERVICE) psql -U postgres -d rag

lab: up
	@echo "Waiting for Postgres to become ready..."
	@until $(COMPOSE) exec -T $(SERVICE) pg_isready -U postgres -d rag >/dev/null 2>&1; do sleep 1; done
	python3 run_lab.py
