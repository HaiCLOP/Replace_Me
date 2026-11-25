"""A chain of models tried left to right ("anthropic:claude-opus-5-5@medium,ollama:my-voice"), with name masking
for every model that isn't on your own PC."""

from __future__ import annotations

from typing import Any

from replace_me.llm.anthropic import EFFORTS, AnthropicClient, RunTool
from replace_me.llm.base import LLMError, LLMResponse
from replace_me.llm.masking import get_masker
from replace_me.llm.ollama import OllamaClient
from replace_me.log import get_logger

logger = get_logger("llm.chain")


def make_client(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"model '{spec}' must look like provider:model")
    prov, model = spec.split(":", 1)
    model, _, effort = model.partition("@")
    prov, effort = prov.strip().lower(), (effort.strip().lower() or None)
    if effort and effort not in EFFORTS:
        raise ValueError(f"effort '{effort}' must be one of {EFFORTS}")
    if prov == "anthropic":
        return AnthropicClient(model.strip(), effort)
    if prov == "ollama":
        if effort:
            raise ValueError("@effort only works on anthropic: models")
        return OllamaClient(model.strip())
    raise ValueError(f"provider '{prov}' not supported (anthropic, ollama); add one in replace_me/llm/")


class Chain:
    def __init__(self, spec: str):
        self.links = [(s.strip(), make_client(s.strip())) for s in spec.split(",") if s.strip()]
        if not self.links:
            raise ValueError("empty model chain")

    def can_use_tools(self) -> bool:
        return any(hasattr(c, "generate_with_tools") for _, c in self.links)

    async def generate(self, prompt: str, system: str | None = None, temperature: float = 0.9,
                       max_tokens: int | None = None) -> LLMResponse:
        masker = get_masker()
        last: Exception | None = None
        for label, client in self.links:
            local = bool(getattr(client, "local", False))
            p, s = (prompt, system) if (local or masker is None) else (masker.mask(prompt), masker.mask(system or "") or None)
            try:
                res = await client.generate(p, s, temperature=temperature, max_tokens=max_tokens)
                if masker is not None and not local:
                    res.content = masker.unmask(res.content)
                return res
            except Exception as e:
                last = e
                logger.warning("llm.failover", model=label, error=str(e)[:160])
        raise last or LLMError("all models failed")

    async def generate_with_tools(self, system: list[dict[str, Any]], messages: list[dict[str, Any]],
                                  tools: list[dict[str, Any]], run_tool: RunTool, max_rounds: int = 2,
                                  max_tokens: int | None = None) -> LLMResponse:
        masker = get_masker()
        if masker is not None:
            system, messages = masker.mask_any(system), masker.mask_any(messages)

        async def tool(name: str, args: dict[str, Any]) -> str:
            out = await run_tool(name, masker.unmask_any(args) if masker else args)
            return masker.mask(out) if masker else out

        last: Exception | None = None
        for label, client in self.links:
            if not hasattr(client, "generate_with_tools"):
                continue
            try:
                res = await client.generate_with_tools(system, messages, tools, tool, max_rounds, max_tokens)
                if masker is not None:
                    res.content = masker.unmask(res.content)
                return res
            except Exception as e:
                last = e
                logger.warning("llm.failover", model=label, tools=True, error=str(e)[:160])
        raise last or LLMError("no model in the chain can use tools")


_chains: dict[str, Chain] = {}


def build_chain(spec: str) -> Chain:
    if spec not in _chains:
        _chains[spec] = Chain(spec)
    return _chains[spec]
