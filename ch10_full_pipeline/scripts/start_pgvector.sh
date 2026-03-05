#!/usr/bin/env bash
set -euo pipefail

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Error: docker compose (or docker-compose) is required." >&2
  exit 1
fi

echo "Starting pgvector service..."
"${COMPOSE_CMD[@]}" up -d pgvector

echo "Waiting for PostgreSQL to become ready..."
for _ in $(seq 1 30); do
  if "${COMPOSE_CMD[@]}" exec -T pgvector pg_isready -U postgres -d text_embeddings >/dev/null 2>&1; then
    echo "pgvector is ready at postgresql://postgres:@localhost:5432/text_embeddings"
    exit 0
  fi
  sleep 1
done

echo "Error: pgvector did not become ready in time." >&2
exit 1
