from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class LLMError(Exception):
    """A model call failed; the chain moves on to the next model."""


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache: dict[str, int] = field(default_factory=dict)  # cache_read / cache_write / input (uncached)
    tool_trace: list[str] = field(default_factory=list)  # lookups the model made
    raw: Any = None
