"""A model on your own PC via Ollama (e.g. your fine-tuned voice, docs/FINE_TUNING.md). Nothing leaves the PC."""

from __future__ import annotations

import re
import time

import httpx

from replace_me.config import get_settings
from replace_me.llm.base import LLMError, LLMResponse

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


class OllamaClient:
    local = True  # gets real names (masking is only for text that leaves the PC)

    def __init__(self, model: str, max_tokens: int = 120):
        self.model, self.max_tokens = model, max_tokens
        self.base_url = get_settings().OLLAMA_BASE_URL.rstrip("/")

    async def generate(self, prompt: str, system: str | None = None, temperature: float = 0.8,
                       max_tokens: int | None = None) -> LLMResponse:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": msgs, "stream": False,
                   "options": {"temperature": temperature, "num_predict": max_tokens or self.max_tokens}}
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                r = await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama not reachable at {self.base_url}: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"Ollama {r.status_code}: {r.text[:200]}")
        data = r.json()
        text = _THINK.sub("", ((data.get("message") or {}).get("content") or "")).strip()
        if not text:
            raise LLMError("Ollama returned no text")
        return LLMResponse(content=text, model=self.model, provider="ollama",
                           latency_ms=(time.perf_counter() - start) * 1000,
                           input_tokens=data.get("prompt_eval_count", 0) or 0,
                           output_tokens=data.get("eval_count", 0) or 0)
