import anthropic

from .base import LLMProvider

# Older models (e.g. Haiku 4.5) don't support adaptive thinking and
# reject the parameter outright — only send it for models that do.
_ADAPTIVE_THINKING_MODELS = {
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-sonnet-5",
}


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8"):
        self.model = model
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Verdicts are long documents — stream to avoid HTTP timeouts,
        # then collect the complete message.
        kwargs = {}
        if self.model in _ADAPTIVE_THINKING_MODELS:
            kwargs["thinking"] = {"type": "adaptive"}

        with self.client.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            **kwargs,
        ) as stream:
            message = stream.get_final_message()
        return "".join(block.text for block in message.content if block.type == "text")
