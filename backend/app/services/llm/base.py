from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface so models from different vendors can be swapped
    and compared side-by-side in the thesis experiments."""

    provider_name: str

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate the verdict text for the given prompts."""
        raise NotImplementedError
