CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1024) NOT NULL
);

-- Optional ANN indexes (create manually when needed):
-- CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
--   ON chunks USING hnsw (embedding vector_ip_ops);
--
-- CREATE INDEX IF NOT EXISTS chunks_embedding_ivfflat
--   ON chunks USING ivfflat (embedding vector_ip_ops) WITH (lists = 100);
