"""Every chat you're in, both sides, as overlapping 10-message chunks: the copy's long-term memory.

Searches are hybrid: meaning (local embeddings) + the literal words (names embed badly: "riya" means nothing to
an embedding model, but it's right there in the text) + a little recency. Also: everything said on a date, the
moment a line was said, and "what was he up to around then" (his own lines in other chats nearby in time).

Live chats (what the bot has seen since the export) are added on top with set_live(), so searches never stop
at the day the export was downloaded.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from replace_me.memory.embeddings import embed
from replace_me.memory.export import Chat

WINDOW, STEP, MAX_CHARS = 10, 6, 900
MIN_SCORE = 0.55
_WORD = re.compile(r"[a-z0-9]{4,}")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _first(name: str) -> str:
    w = re.findall(r"[A-Za-z]+", name or "")
    return w[0] if w else (name or "them")


def build(chats: Sequence[Chat], out_dir: Path) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    for chat in chats:
        who_name = {n: _first(n) for n in chat.names}
        aliases = sorted({a for a in [_norm(chat.folder.rsplit("_", 1)[0])] + [_norm(n) for n in chat.names] if len(a) >= 3})
        msgs = chat.messages
        for start in range(0, len(msgs), STEP):
            win = msgs[start:start + WINDOW]
            if len(win) < 3:
                break
            text = " | ".join(f"{'You' if m.mine else who_name.get(m.sender, _first(m.sender))}: {m.text[:220]}" for m in win)
            rows.append({"chat": chat.folder, "names": chat.names[:12], "aliases": aliases, "group": chat.group,
                         "day": datetime.fromtimestamp(win[0].ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                         "ts": win[-1].ts, "text": text[:MAX_CHARS]})
            if start + WINDOW >= len(msgs):
                break
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chats.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(out_dir / "chats.npy", embed([r["text"] for r in rows]))
    return {"chats": len(chats), "chunks": len(rows)}


class ChatMemory:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.rows: list[dict[str, Any]] = []
        self.vecs: np.ndarray | None = None
        self.by_chat: dict[str, list[int]] = {}
        self.lower: list[str] = []
        self.base_n = 0
        self._live_vecs: dict[str, np.ndarray] = {}
        self._loaded = False

    def load(self) -> "ChatMemory":
        if self._loaded:
            return self
        self._loaded = True
        src, vec = self.data_dir / "chats.jsonl", self.data_dir / "chats.npy"
        if src.exists() and vec.exists():
            self.rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.vecs = np.load(vec)
        else:
            self.vecs = np.zeros((0, 384), dtype=np.float32)
        self.base_n = len(self.rows)
        self._reindex()
        return self

    def _reindex(self) -> None:
        self.by_chat = {}
        for i, r in enumerate(self.rows):
            self.by_chat.setdefault(r["chat"], []).append(i)
        self.lower = [re.sub(r"(.)\1+", r"\1", r["text"].lower()) for r in self.rows]

    # ── live chats on top of the export ─────────────────────────────────────
    def set_live(self, rows: Sequence[dict[str, Any]]) -> int:
        self.load()
        new = [r["text"] for r in rows if r["text"] not in self._live_vecs]
        if new:
            for t, v in zip(new, embed(new)):
                self._live_vecs[t] = v
        self.rows = self.rows[: self.base_n] + list(rows)
        live = [self._live_vecs[r["text"]] for r in rows]
        base = self.vecs[: self.base_n] if self.vecs is not None else np.zeros((0, 384), dtype=np.float32)
        self.vecs = np.vstack([base] + ([np.vstack(live)] if live else []))
        self._reindex()
        return len(rows)

    # ── which chats are with whom ───────────────────────────────────────────
    def chats_for(self, handle: str, name: str = "") -> list[str]:
        """Export chat folder(s) for a live Instagram user (by handle without dots/underscores, or display name)."""
        self.load()
        keys = {k for k in (_norm(handle), _norm(name)) if len(k) >= 3}
        out = []
        for chat, idx in self.by_chat.items():
            r = self.rows[idx[0]]
            if chat.startswith("live:") or r.get("group"):
                continue
            if keys & set(r.get("aliases") or []):
                out.append(chat)
        return out

    def known_names(self) -> set[str]:
        self.load()
        return {re.sub(r"(.)\1+", r"\1", _first(n).lower()) for r in self.rows for n in r.get("names") or []
                if len(_first(n)) >= 3}

    # ── searching ────────────────────────────────────────────────────────────
    def _hybrid(self, idx: Sequence[int], query: str, k: int) -> list[int]:
        idx = list(dict.fromkeys(idx))
        if not idx or self.vecs is None or not len(self.vecs):
            return []
        q = embed([query])[0]
        sub = np.asarray(idx)
        scores = self.vecs[sub] @ q
        terms = [w for w in _WORD.findall(re.sub(r"(.)\1+", r"\1", query.lower()))][:4]
        if terms:
            lex = np.array([sum(t in self.lower[i] for t in terms) / len(terms) for i in sub], dtype=np.float32)
            scores = scores + 0.3 * lex
            if lex.max() > 0:  # a name search that hits: only chunks that contain it
                scores = np.where(lex > 0, scores, scores - 1.0)
        ts_max = max((r["ts"] for r in self.rows), default=1) or 1
        scores = scores + 0.02 * np.array([self.rows[i]["ts"] / ts_max for i in sub], dtype=np.float32)
        return [int(sub[j]) for j in np.argsort(-scores)[:k] if float(scores[j]) >= MIN_SCORE]

    def _fmt(self, hits: Sequence[int], with_names: bool) -> list[str]:
        out = []
        for i in sorted(hits, key=lambda i: self.rows[i]["ts"]):
            r = self.rows[i]
            who = f", chat with {', '.join(r.get('names') or [])[:60]}" if with_names else ""
            out.append(f"[{r['day']}{who}] {r['text']}")
        return list(dict.fromkeys(out))

    def search(self, chats: Sequence[str], query: str, k: int = 4) -> list[str]:
        """Your chat with THIS person only."""
        self.load()
        idx = [i for c in chats for i in self.by_chat.get(c, [])]
        return self._fmt(self._hybrid(idx, query, k), False) if query.strip() else []

    def search_all(self, query: str, k: int = 5) -> list[str]:
        """Every chat (both sides): gossip, what happened with someone."""
        self.load()
        return self._fmt(self._hybrid(range(len(self.rows)), query, k), True)

    def search_mine(self, query: str, k: int = 6) -> list[str]:
        """Only what YOU said, anywhere."""
        out = []
        for chunk in self.search_all(query, k):
            out += [p[5:] for p in chunk.split("] ", 1)[-1].split(" | ") if p.startswith("You: ")]
        return list(dict.fromkeys(out))[:12]

    def on_day(self, month: int, day: int, year: int | None = None, who: str = "", k: int = 6,
               chats: Sequence[str] | None = None) -> list[str]:
        """Everything said on a date (any year if none given); chats with/about `who` first."""
        self.load()
        want = f"-{month:02d}-{day:02d}"
        allowed = set(chats) if chats is not None else None
        idx = [i for i, r in enumerate(self.rows) if r["day"].endswith(want)
               and (year is None or r["day"].startswith(f"{year:04d}-")) and (allowed is None or r["chat"] in allowed)]
        w = re.sub(r"(.)\1+", r"\1", (who or "").lower().strip())

        def rank(i: int) -> tuple[int, int]:
            r = self.rows[i]
            names = " ".join(r.get("names") or []).lower() + " " + " ".join(r.get("aliases") or [])
            return (-(2 if w and w in names else 1 if w and w in self.lower[i] else 0), -r["ts"])

        return self._fmt(sorted(idx, key=rank)[:k], True)

    # ── time ─────────────────────────────────────────────────────────────────
    def when_text(self, text: str, chats: Sequence[str] = ()) -> int | None:
        self.load()
        t = (text or "").strip()
        if len(t) < 4:
            return None
        idx = [i for c in chats for i in self.by_chat.get(c, [])] if chats else range(len(self.rows))
        for i in idx:
            if any(p.split(": ", 1)[-1] == t for p in self.rows[i]["text"].split(" | ")):
                return int(self.rows[i]["ts"])
        return None

    def life_before(self, ts_ms: int, chat: str, exclude: Sequence[str] = (), days: float = 2.0,
                    k_same: int = 30, k_other: int = 50) -> tuple[list[str], list[str]]:
        """What you knew at a moment: (this chat in the day or two before it, your own lines in other chats
        around then, nearest first). Never anything at/after the moment from this chat, never `exclude` lines."""
        self.load()
        span = days * 86400 * 1000
        bad = {t.strip().lower() for t in exclude if t and t.strip()}
        same: list[str] = []
        for i in self.by_chat.get(chat, []):
            r = self.rows[i]
            if ts_ms - span <= int(r["ts"]) < ts_ms - 60_000:
                parts = r["text"].split(" | ")
                if any(p.split(": ", 1)[-1].strip().lower() in bad for p in parts):
                    continue
                same += [p for p in parts if p not in same]
        others: list[tuple[int, str]] = []
        for r in self.rows:
            if r["chat"] == chat or r.get("group") or abs(int(r["ts"]) - ts_ms) > span:
                continue
            for p in r["text"].split(" | "):
                if p.startswith("You: "):
                    t = p[5:].strip()
                    if 4 <= len(t) <= 100 and t.lower() not in bad and not t.startswith("["):
                        others.append((abs(int(r["ts"]) - ts_ms), t))
        seen: list[str] = []
        for _, t in sorted(others):
            if t not in seen:
                seen.append(t)
        return same[-k_same:], seen[:k_other]
