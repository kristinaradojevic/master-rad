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
from .local_mlx_provider import LocalMLXProvider

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

# The fine-tuned model runs locally via MLX (OpenAI's fine-tuning API was
# shut down for self-serve use in 2026 — see README's Fine-tuning section
# and ml/finetune/train_lora.py). It's served by `mlx_lm.server`, which
# speaks the OpenAI chat-completions protocol, but uses LocalMLXProvider
# rather than plain OpenAIProvider — see that module's docstring for why a
# single call isn't enough for this particular model. Only registered once
# LOCAL_MLX_BASE_URL is set, i.e. once the server is actually meant to be
# running.
#
# mlx_lm.server rejects any "model" value in the request body that isn't
# one it recognizes (either the path/repo it was started with, or
# something that resolves to that same path) — it's not just a label, so
# this has to match how the server was launched (see README), not the
# registry key above.
_local_mlx_base_url = os.getenv("LOCAL_MLX_BASE_URL")
_local_mlx_model = os.getenv("LOCAL_MLX_MODEL", "ml/data/finetune/fused_model")
if _local_mlx_base_url:
    _REGISTRY["local-finetuned-qwen"] = {
        "provider": "local-mlx",
        "label": "Qwen 2.5 7B (fine-tuned, local)",
        "factory": lambda: LocalMLXProvider(_local_mlx_model, base_url=_local_mlx_base_url),
        "env_key": "LOCAL_MLX_BASE_URL",
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
