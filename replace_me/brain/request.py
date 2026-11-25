"""The cached request: system blocks + the chat as real user/assistant turns + this turn, with cache breakpoints.

    system   (cache 1 h)   same for every chat
    msg 1    person block  same all chat              } cached together: the breakpoint sits on the last
    turns    the chat      grows                     } turn of the chat so far
    last     now block     this turn only (never cached)

The chat window only moves every 16 messages (aligned_turns): a window sliding by one message would change the
start of the prompt on every reply and miss the cache every time. Measured: ~51k tokens read from cache per
reply, ~4-9k new.
"""

from __future__ import annotations

from typing import Any, Sequence

STEP, MIN_TURNS = 16, 24


def aligned_turns(messages: Sequence[Any]) -> list[Any]:
    n = len(messages)
    if n <= MIN_TURNS + STEP:
        return list(messages)
    return list(messages[((n - MIN_TURNS) // STEP) * STEP:])


def as_turns(history: Sequence[Any], incoming: Sequence[str]) -> list[tuple[str, str]]:
    """Stored messages -> (role, text): theirs = user, yours (sent by the copy or typed by you) = assistant.
    Their trailing messages that are this turn's incoming ones are left out (they come last)."""
    msgs = list(history)
    pending = {m.strip() for m in incoming if m and m.strip()}
    while msgs and pending and getattr(msgs[-1], "who", "") == "them" and msgs[-1].text.strip() in pending:
        msgs.pop()
    turns: list[tuple[str, str]] = []
    for m in msgs:
        role = "user" if m.who == "them" else "assistant"
        text = "\n".join(f"Them: {t}" for t in m.text.splitlines() if t.strip()) if role == "user" else m.text.strip()
        if not text:
            continue
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + "\n" + text)
        else:
            turns.append((role, text))
    return turns


def build(system_text: str, person_block: str, turns: Sequence[tuple[str, str]], now_block: str) -> tuple[list, list]:
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    msgs: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": person_block}]}]
    for role, text in turns:
        if msgs[-1]["role"] == role:
            msgs[-1]["content"].append({"type": "text", "text": text})
        else:
            msgs.append({"role": role, "content": [{"type": "text", "text": text}]})
    msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    if msgs[-1]["role"] == "user":
        msgs[-1]["content"].append({"type": "text", "text": now_block})
    else:
        msgs.append({"role": "user", "content": [{"type": "text", "text": now_block}]})
    return system, msgs
