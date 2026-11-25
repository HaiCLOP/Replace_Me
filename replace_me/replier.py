"""Answer a burst of messages from one person: gather what the brain needs, write the reply, remember it."""

from __future__ import annotations

import time
from typing import Any

from replace_me.brain.lookups import Lookups
from replace_me.brain.reply import Reply, Situation, write_reply
from replace_me.config import get_settings
from replace_me.log import get_logger
from replace_me.memory.chats import ChatMemory
from replace_me.memory.corpus import Corpus
from replace_me.profile import is_bot_aware, life_now, load_profile, master_profile, person, person_note
from replace_me.store import Store
from replace_me.brain.request import aligned_turns

logger = get_logger("replier")


class Replier:
    LIVE_REFRESH_S = 20.0

    def __init__(self, store: Store | None = None):
        s = get_settings()
        self.store = store or Store()
        self.corpus = Corpus(s.DATA_DIR)
        self.memory = ChatMemory(s.DATA_DIR)
        self._live_at = 0.0

    def refresh_live(self) -> None:
        """Chats since the export (live DB) become searchable like the export, in the same 10-message chunks."""
        if time.monotonic() - self._live_at < self.LIVE_REFRESH_S:
            return
        self._live_at = time.monotonic()
        by: dict[str, list[Any]] = {}
        for m in self.store.all_messages():
            by.setdefault(m.handle, []).append(m)
        rows = []
        for handle, msgs in by.items():
            name = self.store.name_of(handle) or handle
            first = name.split()[0] if name.split() else handle
            for start in range(0, len(msgs), 6):
                win = msgs[start:start + 10]
                if len(win) < 2:
                    break
                text = " | ".join(f"{first if m.theirs else 'You'}: {m.text.replace(chr(10), ' / ')[:220]}" for m in win)[:900]
                rows.append({"chat": f"live:{handle}", "names": [name], "aliases": [handle.replace(".", "").replace("_", "")],
                             "group": False, "day": win[0].ts[:10], "ts": _ms(win[-1].ts), "text": text})
                if start + 10 >= len(msgs):
                    break
        self.memory.set_live(rows)

    def _chats(self, handle: str, name: str) -> list[str]:
        return self.memory.chats_for(handle, name) + [f"live:{handle}"]

    def situation(self, handle: str, name: str, incoming: list[str]) -> Situation:
        s = get_settings()
        profile = load_profile()
        self.refresh_live()
        p = person(profile, handle)
        close = is_bot_aware(profile, handle) or int(p.get("score", 0) or 0) >= s.GOSSIP_MIN_SCORE
        chats = self._chats(handle, name)
        export_chats = [c for c in chats if not c.startswith("live:")]
        mem = self.memory
        lk = Lookups(
            them=lambda q: mem.search(chats, q),
            everyone=(lambda q: mem.search_all(q)) if close else None,
            mine=lambda q: mem.search_mine(q),
            day=lambda mo, d, y=None, who="": mem.on_day(mo, d, y, who, chats=None if close else chats),
            examples=lambda q: [e.render() for e in self.corpus.find(q, k=8, n_random=0)],
        )
        if close:
            told = {h: self.store.told_upto(h) for h, _, _ in self.store.others_lately(handle)}
            lk.gossip = [f"@{h} ({self.store.name_of(h)}): {summary}" for h, summary, last in self.store.others_lately(handle)
                         if last > told.get(h, 0)]
        latest = []
        for c in export_chats[:1]:
            for r in [mem.rows[i] for i in mem.by_chat.get(c, [])][-3:]:
                latest += [x[5:] for x in r["text"].split(" | ") if x.startswith("You: ")]
        label = str(p.get("label") or name or handle) + f" (@{handle})"
        return Situation(
            handle=handle, label=label, incoming=incoming,
            history=aligned_turns(self.store.history(handle)),
            person_note=person_note(export_chats), latest=latest[-12:],
            my_days=self.store.my_recent_lines(handle), close=close, clean=bool(p.get("clean")),
            lookups=lk, known_names=sorted(mem.known_names()), gender=str(p.get("gender") or ""),
            bot_aware=is_bot_aware(profile, handle),
        )

    async def answer(self, handle: str, name: str, incoming: list[str], seed: int | None = None) -> Reply:
        self.store.touch(handle, name)
        for m in incoming:
            self.store.add(handle, "them", m)
        sit = self.situation(handle, name, incoming)
        reply = await write_reply(sit, load_profile(), self.corpus, master_profile(), life_now(), seed=seed)
        if reply.bubbles:
            self.store.add(handle, "me", "\n".join(reply.bubbles))
        if reply.told_gossip:
            for h, _, last in self.store.others_lately(handle):
                self.store.mark_told(h, last)
        return reply


def _ms(iso: str) -> int:
    from datetime import datetime

    try:
        return int(datetime.fromisoformat(iso).timestamp() * 1000)
    except ValueError:
        return 0
