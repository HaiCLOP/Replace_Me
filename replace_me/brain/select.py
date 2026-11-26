"""Choosing the reply: a blind-judge-style picker, and option C (your fine-tuned model words every candidate)."""

from __future__ import annotations

import random
import re
from typing import Sequence

from replace_me.config import get_settings
from replace_me.llm.chain import build_chain
from replace_me.llm.masking import get_masker
from replace_me.log import get_logger

logger = get_logger("brain.select")

JUDGE_SYSTEM = (
    "You're an expert at spotting AI imitations of a real person's texting. You get some of his real messages for "
    "reference, the end of a chat, and several replies. Exactly one was sent by the real person; the others were "
    "written by an AI copying him. Pick the REAL one. Judge by how real people text (short, messy, specific, "
    "sometimes off on their own track, not the neatest or most fitting answer), not by which reply is better. "
    "Answer with only the number."
)


async def pick_like_a_judge(options: list[list[str]], tail: Sequence[str], incoming: Sequence[str],
                            your_lines: Sequence[str], rng: random.Random) -> int:
    """Index of the option a blind judge would most likely take for the real you (same view a judge has)."""
    if len(options) < 2:
        return 0
    ref = rng.sample([l for l in your_lines if 3 <= len(l) <= 80], min(15, len(your_lines))) if your_lines else []
    order = list(range(len(options)))
    rng.shuffle(order)
    prompt = ("HIS REAL MESSAGES (reference):\n" + "\n".join(f"- {r}" for r in ref)
              + "\n\nCHAT (newest last):\n" + "\n".join(tail[-6:]) + "\n" + "\n".join(f"Friend: {m}" for m in incoming[-3:])
              + "\n\nREPLIES:\n" + "\n\n".join(f"{i + 1}.\n" + "\n".join(options[o]) for i, o in enumerate(order)))
    try:
        res = await build_chain(get_settings().PICKER_CHAIN).generate(prompt, JUDGE_SYSTEM, temperature=0.0, max_tokens=20)
        m = re.search(r"\d+", res.content or "")
        if m and 1 <= int(m.group()) <= len(order):
            return order[int(m.group()) - 1]
    except Exception as e:
        logger.warning("picker.failed", error=str(e)[:160])
    return 0


# ── option C: your fine-tuned model words what the cloud model decided ────────
# Trained on exactly this shape (tools/finetune/build_dataset.py): a one-line system, the last messages as
# "Them:" / "You:" lines, optionally "MEANING (what you want to say): ...", then "Reply as you:".
VOICE_SYSTEM = ("You are {name}, texting {who} on Instagram from your phone. Reply exactly the way you really do: "
                "your words, spelling, slang, length and emoji. One message per line.")


def voice_prompt(tail: Sequence[str], incoming: Sequence[str], meaning: str) -> str:
    lines = list(tail[-40:]) + [f"Them: {m}" for m in incoming if f"Them: {m}" not in tail[-len(incoming) - 2:]]
    return "\n".join(lines) + (f"\n\nMEANING (what you want to say): {meaning}" if meaning else "") + "\n\nReply as you:"


async def in_your_voice(meaning: list[str], tail: Sequence[str], incoming: Sequence[str], name: str, who: str) -> list[str]:
    """Your fine-tuned model's wording of `meaning`, or [] (not running, or it named someone new)."""
    s = get_settings()
    try:
        res = await build_chain(s.VOICE_MODEL).generate(voice_prompt(tail, incoming, " / ".join(meaning)),
                                                        VOICE_SYSTEM.format(name=name or "me", who=who or "a friend"),
                                                        temperature=0.8, max_tokens=80)
    except Exception as e:
        logger.warning("voice.failed", error=str(e)[:160])
        return []
    lines = [l.strip() for l in (res.content or "").splitlines() if l.strip() and not l.lower().startswith(("them:", "you:", "reply as"))]
    # it learned "[photo]" / "[sent you a reel]" placeholders from the export, but it can't send media
    lines = [l for l in lines if not re.fullmatch(r"\[[^\]]*\]", l)][:5]
    masker = get_masker()
    if lines and masker is not None and masker.names_new_people(" ".join(lines), " ".join(meaning) + " " + " ".join(tail) + " " + who):
        logger.info("voice.new_person_dropped")
        return []  # it may have pulled someone (and their story) from another chat it memorised
    return lines


def split_candidates(raw: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?m)^\s*={3,}\s*$", raw or "") if p.strip()]
    return parts or [raw]


def parse_reply_to(raw: str, theirs: Sequence[str]) -> tuple[str, str]:
    """('REPLY TO: <words>' line removed, the message of theirs it matches best or '')."""
    m = re.search(r"(?im)^\s*reply(?:ing)?\s*to\s*[:\-]\s*(.+)$", raw or "")
    if not m:
        return raw, ""
    want = set(re.findall(r"\w+", m.group(1).lower()))
    best, score = "", 0.0
    for t in theirs:
        have = set(re.findall(r"\w+", t.lower()))
        if want and have:
            s = len(want & have) / len(want)
            if s > score:
                best, score = t, s
    return raw[: m.start()] + raw[m.end():], best if score >= 0.6 else ""
