"""Lookups the model can make while writing (tools), plus the ones always done for it (a date or a person named).

Gossip rule: only people you score >= GOSSIP_MIN_SCORE in me/profile.yaml (or bot_aware_handles) get the
all-chats search and recent gossip; for everyone else those tools don't exist, and date lookups stay inside
their own chat with you.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DM = re.compile(rf"\b(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*(?:of\s+)?{_MON}\b(?:\s*,?\s*(20\d\d))?", re.I)
_MD = re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s*,?\s*(20\d\d))?", re.I)
_NUM = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})(?:[/\-.]((?:20)?\d\d))?\b")
_ISO = re.compile(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b")
STORY_ASK = re.compile(r"\b(?:what\s+happened|tell\s+me|story|gossip|tea|spill|drama|kya\s+hua|batao|sunao|scene|detail)\b", re.I)


def dates_in(text: str) -> list[tuple[int, int, int | None]]:
    """(month, day, year|None) for dates written in a message ('17th April', 'april 17', '17/4', '2026-04-17')."""
    t, found = text or "", []
    for m in _ISO.finditer(t):
        found.append((int(m.group(2)), int(m.group(3)), int(m.group(1))))
    for m in _DM.finditer(t):
        found.append((_MONTHS[m.group(2)[:3].lower()], int(m.group(1)), int(m.group(3)) if m.group(3) else None))
    for m in _MD.finditer(t):
        found.append((_MONTHS[m.group(1)[:3].lower()], int(m.group(2)), int(m.group(3)) if m.group(3) else None))
    for m in _NUM.finditer(_ISO.sub(" ", t)):
        y = m.group(3)
        found.append((int(m.group(2)), int(m.group(1)), (int(y) + 2000 if len(y) == 2 else int(y)) if y else None))
    out: list[tuple[int, int, int | None]] = []
    for mo, d, y in found:
        if 1 <= mo <= 12 and 1 <= d <= 31 and (mo, d, y) not in out:
            out.append((mo, d, y))
    return out[:2]


@dataclass
class Lookups:
    """What this reply may look up. Functions return lists of text chunks."""
    them: Callable[[str], list[str]] | None = None       # their chat with you
    everyone: Callable[[str], list[str]] | None = None   # all chats (close friends only)
    mine: Callable[[str], list[str]] | None = None       # what you said anywhere
    day: Callable[[int, int, int | None, str], list[str]] | None = None
    examples: Callable[[str], list[str]] | None = None   # more of your real replies
    gossip: list[str] = field(default_factory=list)      # other chats lately (close friends only)
    log: list[str] = field(default_factory=list)

    def tools(self) -> list[dict[str, Any]]:
        where = (["this_chat"] if self.them else []) + (["all_chats"] if self.everyone else []) + (["my_messages"] if self.mine else [])
        out: list[dict[str, Any]] = []
        if where:
            out.append({"name": "search_chats", "description": (
                "Search your chat history. this_chat = everything you and this person said; all_chats = every chat "
                "you're in, both sides (gossip, what happened with someone); my_messages = what you yourself said "
                "anywhere. Use a few specific words: a name, the thing, a place. Returns dated chunks."),
                "input_schema": {"type": "object", "properties": {"where": {"type": "string", "enum": where},
                                                                   "query": {"type": "string"}}, "required": ["where", "query"]}})
        if self.day:
            out.append({"name": "chats_on_day", "description": "Everything said on one date (e.g. 2026-04-17 or 17 April).",
                        "input_schema": {"type": "object", "properties": {"date": {"type": "string"}, "person": {"type": "string"}},
                                         "required": ["date"]}})
        if self.examples:
            out.append({"name": "more_of_my_replies", "description": "More of your own real replies to messages like a given one.",
                        "input_schema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}})
        if self.gossip:
            out.append({"name": "recent_gossip", "description": "What's been going on in your other chats lately "
                        "(this person is one of your closest, so you can tell them).",
                        "input_schema": {"type": "object", "properties": {}}})
        return out

    async def run(self, name: str, args: dict[str, Any]) -> str:
        hits: list[str] = []
        if name == "search_chats":
            fn = {"this_chat": self.them, "all_chats": self.everyone, "my_messages": self.mine}.get(str(args.get("where")))
            q = str(args.get("query", ""))[:80]
            hits = fn(q) if fn and q.strip() else []
            self.log.append(f"{args.get('where')}: {q} ({len(hits)})")
        elif name == "chats_on_day" and self.day:
            for mo, d, y in dates_in(str(args.get("date", "")))[:1]:
                hits = self.day(mo, d, y, str(args.get("person", "") or "").lower())
            self.log.append(f"day {args.get('date')} ({len(hits)})")
        elif name == "more_of_my_replies" and self.examples:
            hits = self.examples(str(args.get("message", ""))[:200])
            self.log.append(f"examples ({len(hits)})")
        elif name == "recent_gossip":
            hits = list(self.gossip)
            self.log.append(f"gossip ({len(hits)})")
        return "\n".join(f"- {h}" for h in hits) if hits else "(nothing found)"

    def always(self, incoming: Sequence[str], known_names: Sequence[str]) -> list[str]:
        """Lookups done for sure, whatever the model decides: a date they wrote, a person they named."""
        said = " ".join(incoming[-3:])
        low = re.sub(r"(.)\1+", r"\1", said.lower())
        who = next((n for n in sorted(known_names, key=len, reverse=True) if len(n) >= 3 and re.search(rf"\b{re.escape(n)}", low)), "")
        found = []
        if self.day:
            for mo, d, y in dates_in(said):
                hits = self.day(mo, d, y, who)
                self.log.append(f"day {d}/{mo} ({len(hits)})")
                found.append(f"ON {d}/{mo}{'/' + str(y) if y else ''} (what was said that day):\n" + ("\n".join(f"- {h}" for h in hits) or "- (nothing)"))
        if who:
            for label, fn in (("your chat with them", self.them), ("all your chats", self.everyone)):
                if fn:
                    hits = fn(who)
                    self.log.append(f"{label}: {who} ({len(hits)})")
                    if hits:
                        found.append(f"ABOUT '{who}' ({label}):\n" + "\n".join(f"- {h}" for h in hits))
        return found
