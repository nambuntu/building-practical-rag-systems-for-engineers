from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PipelineProfile:
    name: str
    embedding_model_path: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    backend: str


PROFILES = {
    "cpu_quick": PipelineProfile(
        name="cpu_quick",
        embedding_model_path="BAAI/bge-small-en-v1.5",
        top_k=3,
        chunk_size=320,
        chunk_overlap=32,
        backend="faiss",
    ),
    "cpu_demo": PipelineProfile(
        name="cpu_demo",
        embedding_model_path="BAAI/bge-small-en-v1.5",
        top_k=5,
        chunk_size=256,
        chunk_overlap=40,
        backend="faiss",
    ),
    "cpu_plus": PipelineProfile(
        name="cpu_plus",
        embedding_model_path="BAAI/bge-small-en-v1.5",
        top_k=8,
        chunk_size=300,
        chunk_overlap=50,
        backend="faiss",
    ),
    "pgvector_lab": PipelineProfile(
        name="pgvector_lab",
        embedding_model_path="BAAI/bge-small-en-v1.5",
        top_k=8,
        chunk_size=300,
        chunk_overlap=50,
        backend="pgvector",
    ),
}


def get_profile(name: str) -> PipelineProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {', '.join(sorted(PROFILES))}")
    return PROFILES[name]
