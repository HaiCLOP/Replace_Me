"""Read an Instagram "Download your information" export (JSON) from me/export/.

Expected layout (Instagram's own): me/export/**/inbox/<chat_folder>/message_1.json (more files for long chats).
Each file: {"participants": [{"name": ...}], "messages": [{"sender_name", "timestamp_ms", "content", "share"...}]}.
Instagram double-encodes UTF-8 as latin-1 in these files; fix_mojibake undoes it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

_SYSTEM = [re.compile(p, re.I) for p in (
    r"sent an attachment", r"^reacted ", r"^liked a message", r"started (an audio|a video) call",
    r"^the (audio|video) call ended", r"missed (an audio|a video) call", r"joined the (audio|video) call",
    r"unsent a message", r"^you sent", r"changed the group", r"named the group", r"^https?://\S+$",
)]


def fix_mojibake(text: str) -> str:
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def is_system_line(text: str) -> bool:
    return any(p.search(text) for p in _SYSTEM)


@dataclass
class Message:
    mine: bool
    sender: str
    text: str
    ts: int  # ms


@dataclass
class Chat:
    folder: str
    names: list[str]  # the other participant(s)
    messages: list[Message] = field(default_factory=list)

    @property
    def group(self) -> bool:
        return len(self.names) > 1


def _message_text(m: dict) -> str:
    content = m.get("content")
    share = m.get("share") if isinstance(m.get("share"), dict) else None
    if share and (not content or is_system_line(fix_mojibake(content))):
        caption = re.sub(r"\s+", " ", fix_mojibake(share.get("share_text") or "")).strip()[:120]
        return "[sent you a reel" + (f": {caption}" if caption else "") + "]"
    return fix_mojibake(content or "").strip()


def read_chats(export_dir: Path, my_names: set[str]) -> Iterator[Chat]:
    """Every chat in the export, oldest message first. my_names = how YOUR name appears as sender_name."""
    by_folder: dict[str, list[Path]] = {}
    for f in sorted(export_dir.rglob("message_*.json")):
        by_folder.setdefault(f.parent.name, []).append(f)
    mine = {n.strip() for n in my_names if n and n.strip()}
    for folder, files in sorted(by_folder.items()):
        names: set[str] = set()
        msgs: list[Message] = []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for p in data.get("participants") or []:
                n = fix_mojibake(p.get("name", ""))
                if n and n not in mine:
                    names.add(n)
            for m in data.get("messages") or []:
                text = _message_text(m)
                if not text or is_system_line(text):
                    continue
                sender = fix_mojibake(m.get("sender_name", ""))
                msgs.append(Message(mine=sender in mine, sender=sender, text=text, ts=int(m.get("timestamp_ms", 0))))
        if msgs:
            msgs.sort(key=lambda x: x.ts)
            yield Chat(folder=folder, names=sorted(names), messages=msgs)


def guess_my_name(export_dir: Path) -> str:
    """The sender who appears in the most chats is you."""
    from collections import Counter

    c: Counter[str] = Counter()
    for f in export_dir.rglob("message_1.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for s in {fix_mojibake(m.get("sender_name", "")) for m in data.get("messages") or []}:
            if s:
                c[s] += 1
    return c.most_common(1)[0][0] if c else ""
