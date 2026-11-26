"""Blind test: real you vs your copy, on moments from your own chats.

For each sampled moment the copy sees the chat up to the point where you replied. Your real reply is hidden and
taken out of the examples it is shown, so it can't copy itself. Then a judge model gets both replies in random
order and has to say which one is really you. 50% fooled = it can't tell you apart.

    python -m replace_me eval --n 30 --seed 7          # -> reports/eval_<date>.md
    python -m replace_me eval --n 30 --seed 7 --life   # also give it your life around each moment

Same seed = same moments, so two setups can be compared. With 30 moments, expect +-2-3 of noise between runs:
only trust a difference bigger than that. See docs/EVALUATION.md.
"""

from __future__ import annotations

import random
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from replace_me.brain.lookups import Lookups
from replace_me.brain.reply import Situation, write_reply
from replace_me.config import ROOT, get_settings
from replace_me.llm.chain import build_chain
from replace_me.memory.chats import ChatMemory
from replace_me.memory.corpus import DEFAULT_SWEARS, Corpus, Example
from replace_me.profile import life_now, load_profile, master_profile, person_note
from replace_me.store import Msg

_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

JUDGE_SYSTEM = (
    "You're an expert at spotting AI imitations of a real person's texting. You get some of the real person's "
    "messages for reference, a chat, and two replies to it: one is what the real person actually sent, the other "
    "was written by an AI copying them. Pick the REAL one. Judge by how real people text (short, messy, specific, "
    "sometimes off-topic), not by which reply is better. Answer in exactly this format:\n"
    "REAL: A or B\nTELL: a few words on what gave the AI one away"
)


def stats_of(replies: list[list[str]], swears: str = DEFAULT_SWEARS) -> dict[str, float]:
    msgs = [m for r in replies for m in r]
    if not replies or not msgs:
        return {}
    sw = re.compile(swears, re.I)
    n = len(replies)
    return {
        "messages per reply": statistics.mean(len(r) for r in replies),
        "one-message replies %": 100 * sum(len(r) == 1 for r in replies) / n,
        "words per message": statistics.mean(len(m.split()) for m in msgs),
        "replies with a question %": 100 * sum(any("?" in m for m in r) for r in replies) / n,
        "replies with emoji %": 100 * sum(any(_EMOJI.search(m) for m in r) for r in replies) / n,
        "replies with swearing %": 100 * sum(any(sw.search(m) for m in r) for r in replies) / n,
    }


def squash(s: str) -> str:
    """Compare replies ignoring caps, punctuation and stretched letters ("heyyy" = "hey")."""
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()


def leaks(text: str, hidden: list[str]) -> bool:
    low = text.lower()
    return any(h in low for h in hidden)


async def judge(chain: Any, reference: list[str], ctx: list[str], real: list[str], copy: list[str],
                rng: random.Random) -> tuple[bool, str]:
    """(fooled, tell): fooled = the judge picked the copy as the real one."""
    copy_first = rng.random() < 0.5
    a, b = (copy, real) if copy_first else (real, copy)
    prompt = ("The real person's messages (random, from their chats):\n" + "\n".join(f"- {m}" for m in reference)
              + "\n\nTHE CHAT:\n" + "\n".join(ctx)
              + "\n\nREPLY A:\n" + "\n".join(a) + "\n\nREPLY B:\n" + "\n".join(b))
    res = await chain.generate(prompt, JUDGE_SYSTEM, temperature=0.0, max_tokens=80)
    text = res.content or ""
    m = re.search(r"REAL:\s*([AB])", text)
    tell = (re.search(r"TELL:\s*(.+)", text) or [None, ""])[1].strip()
    return bool(m and m.group(1) == "A") == copy_first, tell


def _moment(ex: Example, memory: ChatMemory, life: bool) -> tuple[list[Msg], list[str], list[str]]:
    """The chat as the copy sees it (history ends with their message), plus your life before it if asked."""
    history = [Msg(id=i, handle=ex.chat, who="them" if t["who"] == "them" else "me", text="\n".join(t["texts"]), ts="")
               for i, t in enumerate(ex.context)]
    before: list[str] = []
    days: list[str] = []
    if life:
        ctx_lines = [x for t in ex.context[-3:] for x in t["texts"]][::-1]
        ts = next((w for w in (memory.when_text(x, [ex.chat]) for x in ctx_lines) if w), None)
        if ts:
            before, days = memory.life_before(ts, ex.chat, exclude=list(ex.reply) + [x for t in ex.context for x in t["texts"]])
    return history, before, days


async def run(n: int = 30, seed: int = 7, life: bool = False, out: Path | None = None) -> Path:
    s = get_settings()
    profile = load_profile()
    swears = str(profile.get("swear_words") or DEFAULT_SWEARS)
    corpus = Corpus(s.DATA_DIR).load()
    memory = ChatMemory(s.DATA_DIR)
    if not corpus.examples:
        raise SystemExit("No replies found: run `python -m replace_me ingest` first.")
    rng = random.Random(seed)
    pool = [e for e in corpus.examples if e.context and e.context[-1]["who"] == "them" and e.context[-1]["texts"]]
    sample = rng.sample(pool, min(n, len(pool)))
    master, now_text = master_profile(), life_now()
    real_find = corpus.find

    rows: list[tuple[Example, list[str], float]] = []
    for i, ex in enumerate(sample, 1):
        hidden = [t.strip().lower() for t in ex.reply if len(t.strip()) >= 3]

        def find(query: str, *a: Any, _held: str = ex.key, **kw: Any) -> list[Example]:
            return [e for e in real_find(query, *a, **{**kw, "exclude_keys": [_held]})]

        corpus.find = find  # type: ignore[method-assign]
        history, before, days = _moment(ex, memory, life)
        note = "\n".join(l for l in person_note([ex.chat]).splitlines() if not leaks(l, hidden))
        incoming = list(ex.context[-1]["texts"])
        sit = Situation(handle=ex.chat, label=ex.chat.rsplit("_", 1)[0], incoming=incoming, history=history,
                        person_note=note, latest=[l for l in before if not leaks(l, hidden)][-12:],
                        my_days=[l for l in days if not leaks(l, hidden)], lookups=Lookups(),
                        bot_aware=True)  # a replayed moment never contains a real "are you a bot"
        t0 = time.perf_counter()
        try:
            reply = await write_reply(sit, profile, corpus, master, now_text, seed=seed + i)
            copy = reply.bubbles
        except Exception as e:  # one failure shouldn't end the run
            copy = [f"(error: {str(e)[:80]})"]
        rows.append((ex, copy, time.perf_counter() - t0))
        print(f"[{i}/{len(sample)}] them: {' / '.join(incoming)[:60]!r}\n   real: {' / '.join(ex.reply)[:70]!r}\n"
              f"   copy: {' / '.join(copy)[:70]!r}  ({rows[-1][2]:.0f}s)", flush=True)
    corpus.find = real_find  # type: ignore[method-assign]

    failed = sum(1 for _, c, _ in rows if c and c[0].startswith("(error"))
    rows = [r for r in rows if not (r[1] and r[1][0].startswith("(error"))]
    real_r = [list(ex.reply) for ex, _, _ in rows]
    copy_r = [c for _, c, _ in rows]

    jchain = build_chain(s.JUDGE_CHAIN)
    jr = random.Random(seed + 999)
    mine = [t for t in corpus.my_lines() if 3 <= len(t) <= 80]
    verdicts: list[tuple[bool, str] | None] = []
    for ex, copy, _ in rows:
        if not copy:
            verdicts.append(None)
            continue
        hidden = {t.strip().lower() for t in ex.reply}
        reference = jr.sample([m for m in mine if m.strip().lower() not in hidden], min(15, len(mine)))
        ctx = [f"{'Friend' if t['who'] == 'them' else 'Them'}: {x}" for t in ex.context[-3:] for x in t["texts"][-3:]]
        try:
            verdicts.append(await judge(jchain, reference, ctx, list(ex.reply), copy, jr))
        except Exception as e:
            print("judge failed:", str(e)[:100])
            verdicts.append(None)

    judged = [v for v in verdicts if v is not None]
    fooled = sum(v[0] for v in judged)
    rs, cs = stats_of(real_r, swears), stats_of([c for c in copy_r if c], swears)
    exact = sum(bool(c) and squash(" ".join(r)) == squash(" ".join(c)) for r, c in zip(real_r, copy_r))
    secs = [t for _, _, t in rows]
    lines = [f"# Real you vs your copy ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"{len(rows)} moments from your chats (seed {seed}{', with your life around each' if life else ''}). "
             "The copy saw each chat up to the moment, with your real reply hidden and removed from its examples."
             + (f" {failed} failed (model errors) and are left out." if failed else ""), "",
             f"Setup: LLM_CHAIN={s.LLM_CHAIN} | CANDIDATES={s.CANDIDATES} | PICKER={s.PICKER or 'off'} | "
             f"USE_TOOLS={s.USE_TOOLS} | VOICE={s.VOICE or 'off'}", "",
             "| | you | copy |", "|---|---|---|"]
    lines += [f"| {k} | {rs[k]:.1f} | {cs.get(k, 0):.1f} |" for k in rs]
    lines += [f"| same reply (ignoring caps/stretching) | | {exact} of {len(rows)} |",
              f"| seconds per reply (median) | | {statistics.median(secs) if secs else 0:.1f} |", ""]
    if judged:
        lines += [f"**Blind judge ({s.JUDGE_CHAIN.split(',')[0]}): fooled {fooled} of {len(judged)} "
                  f"({100 * fooled / len(judged):.0f}%; 50% = can't tell you apart).**", "",
                  "What gave the copy away:", *[f"- {v[1]}" for v in judged if not v[0] and v[1]], ""]
    lines += ["## Side by side", ""]
    for i, (ex, copy, sec) in enumerate(rows, 1):
        ctx = [f"{'Them' if t['who'] == 'them' else 'You'}: {x}" for t in ex.context[-3:] for x in t["texts"][-3:]]
        v = verdicts[i - 1]
        lines += [f"### {i}", "", "```", *ctx, "```", f"- **you:** {' / '.join(ex.reply)}",
                  f"- **copy:** {' / '.join(copy) if copy else '(nothing)'}  _({sec:.0f}s)_"]
        if v is not None:
            lines.append(f"- judge: {'FOOLED' if v[0] else 'spotted the copy'}" + (f" ({v[1]})" if v[1] else ""))
        lines.append("")
    out = out or ROOT / "reports" / f"eval_{datetime.now():%Y%m%d_%H%M}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    if judged:
        print(f"\nBLIND JUDGE: fooled {fooled} of {len(judged)} ({100 * fooled / len(judged):.0f}%; 50% = indistinguishable)")
    print(f"report: {out}")
    return out
