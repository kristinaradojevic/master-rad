from openai import OpenAI

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, model: str, base_url: str | None = None):
        self.model = model
        # base_url lets this same provider talk to any OpenAI-compatible
        # endpoint — used for the local MLX server serving the fine-tuned
        # model (see llm/registry.py), which needs no real API key.
        self.client = OpenAI(base_url=base_url, api_key="not-needed" if base_url else None)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
