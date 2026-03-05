from __future__ import annotations

from ollama import chat as ollama_chat


class OllamaChatProvider:
    """Standalone provider duplicated from ch-final with local adaptation."""

    def chat(self, prompt: str, *, model: str) -> str:
        response = ollama_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return response.message.content

    def chat_with_metadata(self, prompt: str, *, model: str) -> dict:
        response = ollama_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        usage = {
            "prompt_tokens": getattr(response, "prompt_eval_count", None),
            "completion_tokens": getattr(response, "eval_count", None),
        }
        if usage["prompt_tokens"] is not None and usage["completion_tokens"] is not None:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return {
            "text": response.message.content,
            "usage": usage,
        }
