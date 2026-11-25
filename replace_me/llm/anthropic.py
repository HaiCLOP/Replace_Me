"""Anthropic Messages API: plain calls, and conversations with lookup tools and prompt caching.

Notes that cost real debugging time:
  - Claude 5.x models reject a custom temperature; with "@effort" they think (adaptive) at that depth.
  - Opus 5.5 always thinks: never send thinking=disabled to it.
  - Thinking shares max_tokens with the answer: a 350-token cap starves it, so thinking requests get >= 4000.
  - After a tool round Opus sometimes ends with no text at all; it's asked once more ("now write your reply").
  - Thinking blocks must be sent back unchanged in the next turn of a tool conversation.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Awaitable, Callable

import httpx

from replace_me.config import get_settings
from replace_me.llm.base import LLMError, LLMResponse
from replace_me.log import get_logger

logger = get_logger("llm.anthropic")

VERSION = "2023-06-01"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
THINKING_MIN_TOKENS = 4000
_NO_SAMPLING = re.compile(r"claude-(?:opus|sonnet|haiku)-5|claude-opus-4-[78]", re.I)
_ALWAYS_THINKS = re.compile(r"claude-opus-5-5", re.I)

RunTool = Callable[[str, dict[str, Any]], Awaitable[str]]


def model_params(model: str, temperature: float, effort: str | None) -> dict[str, Any]:
    m = (model or "").lower()
    params: dict[str, Any] = {}
    if not _NO_SAMPLING.search(m):
        params["temperature"] = max(0.0, min(1.0, float(temperature)))
    if effort in EFFORTS:
        if not _ALWAYS_THINKS.search(m):
            params["thinking"] = {"type": "adaptive"}
        params["output_config"] = {"effort": effort}
        params.pop("temperature", None)
    elif "claude-sonnet-5-5" in m:
        params["thinking"] = {"type": "between_tools"}
    elif re.search(r"claude-haiku-5-5|claude-opus-4-[78]", m):
        params["thinking"] = {"type": "disabled"}
    return params


def thinks(model: str, effort: str | None) -> bool:
    return bool(effort) or bool(_ALWAYS_THINKS.search(model or ""))


class AnthropicClient:
    local = False
    transport: httpx.AsyncBaseTransport | None = None  # tests inject a mock

    def __init__(self, model: str, effort: str | None = None, max_tokens: int = 350):
        s = get_settings()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.api_key = s.ANTHROPIC_API_KEY
        self.base_url = s.ANTHROPIC_BASE_URL.rstrip("/")
        self.timeout = s.LLM_TIMEOUT_SECONDS

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": VERSION, "content-type": "application/json"}

    def _payload(self, system: Any, messages: list[dict[str, Any]], max_tokens: int | None, temperature: float) -> dict[str, Any]:
        cap = max_tokens or self.max_tokens
        p: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(cap, THINKING_MIN_TOKENS) if thinks(self.model, self.effort) else cap,
            "messages": messages,
        }
        if system:
            p["system"] = system
        p.update(model_params(self.model, temperature, self.effort))
        return p

    async def _post(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await client.post(f"{self.base_url}/v1/messages", headers=self._headers(), json=payload)
        except httpx.HTTPError as e:
            raise LLMError(f"request failed: {e}") from e
        if r.status_code == 400 and "thinking" in payload and "thinking" in r.text.lower():
            payload.pop("thinking", None)  # this model/gateway doesn't take our thinking setting
            r = await client.post(f"{self.base_url}/v1/messages", headers=self._headers(), json=payload)
        if r.status_code != 200:
            raise LLMError(f"Anthropic {r.status_code}: {r.text[:300]}")
        return r.json()

    def _timeout(self) -> float:
        return max(self.timeout, 60.0) if thinks(self.model, self.effort) else self.timeout

    async def generate(self, prompt: str, system: str | None = None, temperature: float = 0.9,
                       max_tokens: int | None = None) -> LLMResponse:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout(), transport=self.transport) as client:
            data = await self._post(client, self._payload(system, [{"role": "user", "content": prompt}], max_tokens, temperature))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        if not text.strip():
            raise LLMError(f"no text (stop_reason={data.get('stop_reason')})")
        u = data.get("usage") or {}
        return LLMResponse(content=text, model=data.get("model", self.model), provider="anthropic",
                           latency_ms=(time.perf_counter() - start) * 1000, input_tokens=u.get("input_tokens", 0) or 0,
                           output_tokens=u.get("output_tokens", 0) or 0)

    async def generate_with_tools(self, system: list[dict[str, Any]], messages: list[dict[str, Any]],
                                  tools: list[dict[str, Any]], run_tool: RunTool, max_rounds: int = 2,
                                  max_tokens: int | None = None, temperature: float = 0.9) -> LLMResponse:
        """A conversation where the model may call tools (each result is sent back) up to max_rounds times,
        then must answer. system/messages may carry cache_control breakpoints (see brain/request.py)."""
        msgs = list(messages)
        trace: list[str] = []
        usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        nudged = False
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout(), transport=self.transport) as client:
            for i in range(max_rounds + 2):
                rnd = min(i, max_rounds)
                payload = self._payload(system, msgs, max_tokens, temperature)
                payload["tools"] = tools
                if rnd == max_rounds:
                    payload["tool_choice"] = {"type": "none"}  # enough looking things up: answer now
                data = await self._post(client, payload)
                u = data.get("usage") or {}
                usage["input"] += u.get("input_tokens", 0) or 0
                usage["output"] += u.get("output_tokens", 0) or 0
                usage["cache_read"] += u.get("cache_read_input_tokens", 0) or 0
                usage["cache_write"] += u.get("cache_creation_input_tokens", 0) or 0
                content = data.get("content") or []
                calls = [b for b in content if b.get("type") == "tool_use"]
                if data.get("stop_reason") == "tool_use" and calls and rnd < max_rounds:
                    msgs.append({"role": "assistant", "content": content})
                    results = []
                    for call in calls:
                        args = call.get("input") or {}
                        try:
                            out = await run_tool(call.get("name", ""), args)
                        except Exception as e:  # a broken lookup must not lose the reply
                            out = f"(lookup failed: {str(e)[:120]})"
                        trace.append(f"{call.get('name')}({json.dumps(args, ensure_ascii=False)[:100]})")
                        results.append({"type": "tool_result", "tool_use_id": call["id"],
                                        "content": (out or "(nothing found)")[:6000]})
                    msgs.append({"role": "user", "content": results})
                    continue
                text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
                if not text.strip() and trace and not nudged:
                    nudged = True
                    if content:
                        msgs.append({"role": "assistant", "content": content})
                        msgs.append({"role": "user", "content": "Now write your reply: only your message(s)."})
                    continue
                if not text.strip():
                    raise LLMError(f"no text (stop_reason={data.get('stop_reason')})")
                logger.info("llm.tools_done", model=self.model, lookups=len(trace), **usage)
                return LLMResponse(content=text, model=data.get("model", self.model), provider="anthropic",
                                   latency_ms=(time.perf_counter() - start) * 1000,
                                   input_tokens=usage["input"] + usage["cache_read"] + usage["cache_write"],
                                   output_tokens=usage["output"], cache=usage, tool_trace=trace)
        raise LLMError("tool loop ended without an answer")
