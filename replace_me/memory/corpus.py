"""Your real replies: every "they said X -> you replied Y" from your export, searchable by meaning.

These are the strongest style signal the copy gets: for each incoming message it sees your real replies to the
most similar messages you've ever received. Also measures how you text (reply length, burst size, emoji,
questions, swearing) so the pipeline can hold the copy to YOUR numbers.

    .data/corpus.jsonl   one exchange per line
    .data/corpus.npy     embedding of each exchange's incoming message
    .data/stats.json     your measured habits
"""

from __future__ import annotations

import collections
import json
import random
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from replace_me.memory.embeddings import embed
from replace_me.memory.export import Chat

TURN_GAP_MS = 15 * 60 * 1000   # same-sender messages closer than this are one turn
REPLY_GAP_MS = 45 * 60 * 1000  # your turn must follow theirs within this to count as a reply
CONTEXT_TURNS = 4
MAX_BUBBLES = 5
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_PRIVATE = [re.compile(r"\d{6,}"), re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)]
DEFAULT_SWEARS = r"\b(fuck\w*|shit\w*|wtf|bitch\w*|bc|mc|bkl|bsdk|chutiya\w*|madarchod|behenchod|gandu|bhosd\w*)\b"


@dataclass
class Example:
    chat: str
    context: list[dict[str, Any]]  # [{"who": "them"|"me", "texts": [...]}] oldest first
    reply: list[str]
    key: str

    def render(self, turns: int = 2) -> str:
        lines = [f"{'Them' if t['who'] == 'them' else 'You'}: {x}" for t in self.context[-turns:] for x in t["texts"][-3:]]
        return "\n".join(lines + [f"You: {r}" for r in self.reply])


def _scrub(text: str) -> str:
    for p in _PRIVATE:
        text = p.sub("[number]" if p is _PRIVATE[0] else "[email]", text)
    return text


def _turns(chat: Chat) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in chat.messages:
        who = "me" if m.mine else "them"
        if out and out[-1]["who"] == who and m.ts - out[-1]["end"] <= TURN_GAP_MS:
            out[-1]["texts"].append(m.text)
            out[-1]["end"] = m.ts
        else:
            out.append({"who": who, "texts": [m.text], "start": m.ts, "end": m.ts})
    return out


def build(chats: Sequence[Chat], out_dir: Path, swear_pattern: str = DEFAULT_SWEARS) -> dict[str, Any]:
    examples: list[Example] = []
    mine_all: list[str] = []
    bursts: list[int] = []
    for chat in chats:
        if chat.group:
            continue  # group chats teach a different register; your 1:1 replies are the voice
        turns = _turns(chat)
        for i, t in enumerate(turns):
            if t["who"] != "me":
                continue
            mine_all += t["texts"]
            if i == 0 or turns[i - 1]["who"] != "them" or t["start"] - turns[i - 1]["end"] > REPLY_GAP_MS:
                continue
            reply = [x[:160] for x in t["texts"][:MAX_BUBBLES]]
            if any(p.search(" ".join(reply)) for p in _PRIVATE):
                continue
            bursts.append(len(reply))
            # their side is shown to the model as an example too: their numbers and emails never go in
            ctx = [{"who": c["who"], "texts": [_scrub(x) for x in c["texts"][-4:]]} for c in turns[max(0, i - CONTEXT_TURNS):i]]
            key = _scrub(" ".join(turns[i - 1]["texts"][-3:]))[:300]
            examples.append(Example(chat=chat.folder, context=ctx, reply=reply, key=key))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
    np.save(out_dir / "corpus.npy", embed([e.key for e in examples]))
    stats = measure(mine_all, bursts, [sum(len(r.split()) for r in e.reply) for e in examples], swear_pattern)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"examples": len(examples), **{k: stats[k] for k in ("messages", "swear_rate", "median_words")}}


def measure(mine: Sequence[str], bursts: Sequence[int], reply_words: Sequence[int], swear_pattern: str) -> dict[str, Any]:
    n = max(len(mine), 1)
    sw = re.compile(swear_pattern, re.I)
    c = collections.Counter(bursts)
    lb = collections.Counter(0 if w <= 1 else 1 if w <= 5 else 2 if w <= 12 else 3 for w in reply_words)
    words = collections.Counter(re.sub(r"(.)\1{2,}", r"\1\1", w) for m in mine for w in re.findall(r"[a-zA-Z]{2,}", m.lower()))
    return {
        "messages": len(mine),
        "burst_sizes": {str(k): round(v / max(len(bursts), 1), 3) for k, v in sorted(c.items())},
        "length_mix": [round(lb.get(i, 0) / max(len(reply_words), 1), 3) for i in range(4)],  # 1 word, 2-5, 6-12, 13+
        "emoji_rate": round(sum(bool(_EMOJI.search(m)) for m in mine) / n, 3),
        "question_rate": round(sum("?" in m for m in mine) / n, 3),
        "swear_rate": round(sum(bool(sw.search(m)) for m in mine) / n, 4),
        "capital_start_rate": round(sum(m[:1].isupper() for m in mine) / n, 3),
        "median_words": statistics.median([len(m.split()) for m in mine]) if mine else 0,
        "top_words": [w for w, _ in words.most_common(60)],
    }


class Corpus:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.examples: list[Example] = []
        self.vecs: np.ndarray | None = None
        self.stats: dict[str, Any] = {}
        self._loaded = False

    def load(self) -> "Corpus":
        if not self._loaded:
            self._loaded = True
            src = self.data_dir / "corpus.jsonl"
            if src.exists():
                self.examples = [Example(**json.loads(l)) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
                self.vecs = np.load(self.data_dir / "corpus.npy")
                st = self.data_dir / "stats.json"
                self.stats = json.loads(st.read_text(encoding="utf-8")) if st.exists() else {}
        return self

    def find(self, query: str, k: int = 16, n_random: int = 2, avoid: Sequence[str] = (), seed: int | None = None,
             exclude_keys: Sequence[str] = ()) -> list[Example]:
        """Your replies to the k most similar messages, plus a couple of random ones (so it doesn't over-fit)."""
        self.load()
        if not self.examples or self.vecs is None:
            return []
        q = embed([query or "hi"])[0]
        order = np.argsort(-(self.vecs @ q))
        avoid_set = {a.strip().lower() for a in avoid}
        skip = set(exclude_keys)
        picked: list[Example] = []
        for j in order:
            e = self.examples[int(j)]
            if e.key in skip or any(r.strip().lower() in avoid_set for r in e.reply):
                continue
            picked.append(e)
            if len(picked) >= k:
                break
        rng = random.Random(seed)
        pool = [e for e in self.examples if e not in picked and e.key not in skip]
        return picked + rng.sample(pool, min(n_random, len(pool)))

    def my_lines(self) -> list[str]:
        self.load()
        return [r for e in self.examples for r in e.reply]
