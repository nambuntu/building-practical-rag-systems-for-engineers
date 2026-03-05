from domain.interfaces import EmbeddingProvider
from domain.types import SentenceRecord


def build_sentence_records(
    file_name: str,
    sentences: list[str],
    embeddings: list[list[float]],
) -> list[SentenceRecord]:
    if len(sentences) != len(embeddings):
        raise ValueError("Sentences and embeddings lengths do not match.")

    return [
        SentenceRecord(
            file_name=file_name,
            sentence_number=index + 1,
            content=sentence,
            embedding=embedding,
        )
        for index, (sentence, embedding) in enumerate(zip(sentences, embeddings))
    ]


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def embed_sentences(self, sentences: list[str]) -> list[list[float]]:
        return self.provider.embed_texts(sentences)

    def embed_query(self, query: str) -> list[float]:
        return self.provider.embed_query(query)
