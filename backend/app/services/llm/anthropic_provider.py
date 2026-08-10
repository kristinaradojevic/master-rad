import anthropic

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8"):
        self.model = model
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Verdicts are long documents — stream to avoid HTTP timeouts,
        # then collect the complete message.
        with self.client.messages.stream(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            message = stream.get_final_message()
        return "".join(block.text for block in message.content if block.type == "text")
