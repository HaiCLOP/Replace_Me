"""The live side: every message the bot sees or sends, in .data/live.db (SQLite, local only).

who: 'them' (they sent it), 'me' (your copy sent it), 'manual' (you typed it yourself from your phone: learnt when
the bot opens the chat, so the copy knows what you said too).
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from replace_me.config import get_settings

_DDL = """
CREATE TABLE IF NOT EXISTS conversations (handle TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, handle TEXT NOT NULL, who TEXT NOT NULL, text TEXT NOT NULL, ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_chat ON messages(handle, id);
CREATE TABLE IF NOT EXISTS muted (handle TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS gossip_told (handle TEXT PRIMARY KEY, upto INTEGER NOT NULL);
"""


@dataclass
class Msg:
    id: int
    handle: str
    who: str
    text: str
    ts: str

    @property
    def theirs(self) -> bool:
        return self.who == "them"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (s or "").lower())).strip()


class Store:
    _lock = threading.Lock()

    def __init__(self, path: Path | None = None):
        self.path = path or get_settings().db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_DDL)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _rows(self, sql: str, args: Sequence = ()) -> list[sqlite3.Row]:
        with self._lock, self._conn() as c:
            return c.execute(sql, args).fetchall()

    def _exec(self, sql: str, args: Sequence = ()) -> None:
        with self._lock, self._conn() as c:
            c.execute(sql, args)

    # ── conversations ────────────────────────────────────────────────────────
    def touch(self, handle: str, name: str = "") -> None:
        self._exec("INSERT INTO conversations(handle, name) VALUES(?, ?) ON CONFLICT(handle) DO UPDATE SET "
                   "name = CASE WHEN excluded.name != '' THEN excluded.name ELSE conversations.name END", (handle, name))

    def people(self) -> list[tuple[str, str]]:
        return [(r["handle"], r["name"]) for r in self._rows("SELECT handle, name FROM conversations")]

    def name_of(self, handle: str) -> str:
        r = self._rows("SELECT name FROM conversations WHERE handle = ?", (handle,))
        return r[0]["name"] if r else ""

    # ── messages ─────────────────────────────────────────────────────────────
    def add(self, handle: str, who: str, text: str) -> None:
        if text and text.strip():
            self._exec("INSERT INTO messages(handle, who, text, ts) VALUES(?, ?, ?, ?)", (handle, who, text.strip(), _now()))

    def drop_last_mine(self, handle: str) -> None:
        """A reply that never got sent must not stay in the history as if it was said."""
        self._exec("DELETE FROM messages WHERE id = (SELECT MAX(id) FROM messages WHERE handle = ? AND who = 'me')", (handle,))

    def history(self, handle: str, limit: int = 2000) -> list[Msg]:
        rows = self._rows("SELECT * FROM messages WHERE handle = ? ORDER BY id DESC LIMIT ?", (handle, limit))
        return [Msg(r["id"], r["handle"], r["who"], r["text"], r["ts"]) for r in reversed(rows)]

    def all_messages(self) -> list[Msg]:
        return [Msg(r["id"], r["handle"], r["who"], r["text"], r["ts"]) for r in self._rows("SELECT * FROM messages ORDER BY handle, id")]

    def learn_manual(self, handle: str, my_bubbles: Sequence[str]) -> int:
        """Your own bubbles on screen after the last one the bot sent, that it never sent: you typed them."""
        bubbles = [b for b in my_bubbles if b and b.strip() and not b.strip().startswith("[")]
        known = " \n ".join(_norm(m.text) for m in self.history(handle, 80) if m.who in ("me", "manual"))
        if not bubbles or not known:
            return 0
        anchor = None
        for i, b in enumerate(bubbles):
            if _norm(b) and _norm(b) in known:
                anchor = i
        if anchor is None:
            return 0  # none of the bot's own messages on screen: don't flood the history with old ones
        new = [b.strip() for b in bubbles[anchor + 1:] if _norm(b) and _norm(b) not in known]
        if new:
            self.add(handle, "manual", "\n".join(new))
        return len(new)

    def my_recent_lines(self, except_handle: str, days: float = 4.0, k: int = 40) -> list[str]:
        """What you typed yourself in your other chats lately: real things in your life right now."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._rows("SELECT text FROM messages WHERE who = 'manual' AND handle != ? AND ts >= ? ORDER BY id DESC LIMIT 200",
                          (except_handle, since))
        out: list[str] = []
        for r in rows:
            for line in r["text"].splitlines():
                if 4 <= len(line.strip()) <= 100 and line.strip() not in out:
                    out.append(line.strip())
        return out[:k]

    def others_lately(self, except_handle: str, hours: float = 72, k: int = 4) -> list[tuple[str, str, int]]:
        """(handle, last few lines, newest message id) for your other chats active lately: gossip material."""
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self._rows("SELECT handle, MAX(id) AS last FROM messages WHERE who = 'them' AND handle != ? AND ts >= ? "
                          "GROUP BY handle ORDER BY last DESC LIMIT ?", (except_handle, since, k))
        out = []
        for r in rows:
            lines = [f"{'they' if m.theirs else 'you'}: {m.text.replace(chr(10), ' / ')[:90]}" for m in self.history(r["handle"], 8)]
            out.append((r["handle"], " | ".join(lines)[:500], int(r["last"])))
        return out

    def told_upto(self, handle: str) -> int:
        r = self._rows("SELECT upto FROM gossip_told WHERE handle = ?", (handle,))
        return int(r[0]["upto"]) if r else 0

    def mark_told(self, handle: str, upto: int) -> None:
        self._exec("INSERT INTO gossip_told(handle, upto) VALUES(?, ?) ON CONFLICT(handle) DO UPDATE SET upto = excluded.upto",
                   (handle, upto))

    # ── replies off for some people ──────────────────────────────────────────
    def mute(self, handle: str, name: str = "", on: bool = True) -> None:
        if on:
            self._exec("INSERT OR REPLACE INTO muted(handle, name) VALUES(?, ?)", (handle.lower().lstrip("@"), name))
        else:
            self._exec("DELETE FROM muted WHERE handle = ?", (handle.lower().lstrip("@"),))

    def is_muted(self, handle: str = "", name: str = "") -> bool:
        h = (handle or "").lower().lstrip("@")
        if h and h in get_settings().ignored():
            return True
        for r in self._rows("SELECT handle, name FROM muted"):
            if (h and r["handle"] == h) or (name and r["name"] and r["name"].strip().lower() == name.strip().lower()):
                return True
        return False
