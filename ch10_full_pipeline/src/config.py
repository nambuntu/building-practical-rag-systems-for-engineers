from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:@localhost/text_embeddings"
    embedding_model_path: str = "BAAI/bge-small-en-v1.5"
    chat_model: str = "qwen3:0.6b"

    runs_dir: str = "runs"
    vector_backend: str = "faiss"
    profile: str = "cpu_demo"

    top_k: int = 5
    context_window: int = 2

    squad_default_split: str = "train"
    squad_default_sample_size: int = 5000
    squad_eval_split: str = "validation"
    squad_eval_max_queries: int = 300

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
