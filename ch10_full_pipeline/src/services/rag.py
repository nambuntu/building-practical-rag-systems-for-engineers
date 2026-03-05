from domain.interfaces import ChatProvider
from domain.types import ContextWindow, RagResult
from services.prompting import build_prompt
from services.retrieval import RetrievalService


class RagService:
    def __init__(self, retrieval_service: RetrievalService, chat_provider: ChatProvider) -> None:
        self.retrieval_service = retrieval_service
        self.chat_provider = chat_provider

    def retrieve_context(self, query: str, *, top_k: int, context_window: int) -> list[ContextWindow]:
        return self.retrieval_service.retrieve_context(
            query=query,
            top_k=top_k,
            context_window=context_window,
        )

    def build_prompt(self, context_windows: list[ContextWindow], query: str) -> str:
        return build_prompt(context_windows=context_windows, query=query)

    def generate_answer(self, prompt: str, *, model: str | None = None) -> str:
        return self.chat_provider.chat(prompt=prompt, model=model)

    def answer_query(self, query: str, *, top_k: int, context_window: int, model: str | None = None) -> RagResult:
        contexts = self.retrieve_context(query=query, top_k=top_k, context_window=context_window)
        prompt = self.build_prompt(context_windows=contexts, query=query)
        answer = self.generate_answer(prompt=prompt, model=model)
        return RagResult(query=query, prompt=prompt, answer=answer, contexts=contexts)
