"""Model clients: Anthropic (cache + tools + thinking), Ollama (local), a fallback chain, and name masking."""

from replace_me.llm.base import LLMError, LLMResponse
from replace_me.llm.chain import Chain, build_chain

__all__ = ["Chain", "LLMError", "LLMResponse", "build_chain"]
