"""Model registry — the single place to add or remove models being compared.

Each entry maps a stable key (used by the frontend and stored with results)
to a factory. A model is only offered if its provider's API key is set,
so you can run with any subset of providers configured.
"""

import os
from functools import lru_cache

from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider

_REGISTRY: dict[str, dict] = {
    "gpt-4o": {
        "provider": "openai",
        "label": "GPT-4o",
        "factory": lambda: OpenAIProvider("gpt-4o"),
        "env_key": "OPENAI_API_KEY",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "label": "GPT-4o mini",
        "factory": lambda: OpenAIProvider("gpt-4o-mini"),
        "env_key": "OPENAI_API_KEY",
    },
    "gpt-5.6-sol": {
        "provider": "openai",
        "label": "GPT-5.6 Sol",
        "factory": lambda: OpenAIProvider("gpt-5.6-sol"),
        "env_key": "OPENAI_API_KEY",
    },
    "gpt-5.6-terra": {
        "provider": "openai",
        "label": "GPT-5.6 Terra",
        "factory": lambda: OpenAIProvider("gpt-5.6-terra"),
        "env_key": "OPENAI_API_KEY",
    },
    "gpt-5.6-luna": {
        "provider": "openai",
        "label": "GPT-5.6 Luna",
        "factory": lambda: OpenAIProvider("gpt-5.6-luna"),
        "env_key": "OPENAI_API_KEY",
    },
    "claude-opus-4-8": {
        "provider": "anthropic",
        "label": "Claude Opus 4.8",
        "factory": lambda: AnthropicProvider("claude-opus-4-8"),
        "env_key": "ANTHROPIC_API_KEY",
    },
    "claude-opus-5": {
        "provider": "anthropic",
        "label": "Claude Opus 5",
        "factory": lambda: AnthropicProvider("claude-opus-5"),
        "env_key": "ANTHROPIC_API_KEY",
    },
    "claude-sonnet-5": {
        "provider": "anthropic",
        "label": "Claude Sonnet 5",
        "factory": lambda: AnthropicProvider("claude-sonnet-5"),
        "env_key": "ANTHROPIC_API_KEY",
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "label": "Claude Haiku 4.5",
        "factory": lambda: AnthropicProvider("claude-haiku-4-5"),
        "env_key": "ANTHROPIC_API_KEY",
    },
}


def available_models() -> list[dict]:
    return [
        {"key": key, "provider": entry["provider"], "label": entry["label"]}
        for key, entry in _REGISTRY.items()
        if os.getenv(entry["env_key"])
    ]


@lru_cache(maxsize=None)
def get_provider(key: str) -> LLMProvider:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model: {key}")
    return _REGISTRY[key]["factory"]()
