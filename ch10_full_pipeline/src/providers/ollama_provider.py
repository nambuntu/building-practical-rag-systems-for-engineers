from ollama import chat

from config import Settings
from domain.interfaces import ChatProvider


class OllamaChatProvider(ChatProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chat(self, prompt: str, *, model: str | None = None) -> str:
        selected_model = model or self.settings.chat_model
        response = chat(
            model=selected_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return response.message.content
