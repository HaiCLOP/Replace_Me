"""Real names never leave your PC: swapped for stable codes before a prompt goes to a cloud model, swapped back after.

Everything sent (system prompt, chat, your notes, lookup results) has each person's name and Instagram handle
replaced by a code like "P4KZ" / "@H9QA". The code comes from a hash of the name, so it's the same in every
request (prompt caching keeps working) and stretched spellings map to the same code ("Riyaaa" = "Riya").
Tool inputs the model sends back are unmasked before they run locally; replies are unmasked before sending.

Where names come from (all local): people in your chat export, people the bot has talked to, the people listed
in your notes on each person, the names in me/profile.yaml, and me/mask_extra.txt (me/mask_keep.txt = never mask).
Single words that are among the most common words in your chats are never masked unless they come from a sure
source (a chat participant): "night" in a display name like "Good Night" must not become a code.

Measured on a real export: ~200 names, about 3-4% of words become codes, no ordinary words masked, and the
blind-judge score didn't drop (docs/EVALUATION.md).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from replace_me.log import get_logger

logger = get_logger("llm.masking")

_STOP = {
    "sir", "mam", "maam", "didi", "bhaiya", "bhai", "uncle", "aunty", "group", "girls", "boys", "girl", "boy", "mom",
    "mother", "dad", "father", "papa", "mummy", "teacher", "family", "friends", "friend", "school", "class", "the",
    "and", "his", "her", "their", "other", "others", "brother", "sister", "cousin", "crush", "best", "close", "old",
    "new", "instagram", "insta", "nobody", "somebody", "someone", "user", "unknown", "also", "with", "him", "them",
    "you", "they", "who", "what", "people", "person", "name", "tuition", "coaching", "account", "team", "squad",
}
_ENGLISH = set("""
the and you are for not but can all one out get who how now new see him her day way man boy say she may use two
was has had have his its our their them they this that with from your what when where why will would could should
been being just like love good night morning bro sis dude girl guy king queen god sun star moon rose angel baby
honey sweet cute pretty nice cool best real fake life time world home team game play song music dance movie happy
sad mad crazy funny smart dark light black white red blue green pink gold silver iron little big small miss lady
prince princess boss hero devil ghost storm fire ice snow rain sky ocean
""".split())
_WORLD = re.compile(r"(?ims)^##[^\n]*world[^\n]*\n(.*?)(?=^##\s)")
_BOLD_LABEL = re.compile(r"^\s*[-*]\s+\*\*\"?([^*:\"]{2,80})\"?:?\*\*", re.M)
_TITLE = re.compile(r"^#\s+(.+)$", re.M)
_CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_KEEP_KEYS = ("type", "id", "tool_use_id", "name", "role", "cache_control", "signature")


def _squash(w: str) -> str:
    return re.sub(r"(.)\1+", r"\1", w.lower())


def _clean_words(raw: str) -> list[str]:
    out = []
    for w in re.split(r"[^A-Za-z]+", raw or ""):
        w = re.sub(r"(.)\1+$", r"\1", w)
        if len(w) >= 3 and w.lower() not in _STOP:
            out.append(w[:1].upper() + w[1:].lower())
    return out


def _code(kind: str, key: str) -> str:
    n = int(hashlib.sha1(f"{kind}:{key}".encode()).hexdigest()[:10], 16)
    s = "".join(_CODE_CHARS[(n >> (5 * i)) & 31] for i in range(3))
    return ("P" if kind == "name" else "@H") + s


class NameMasker:
    def __init__(self, names: Iterable[str] = (), handles: Iterable[str] = (), keep: Iterable[str] = ()):
        keep_l = {k.lower() for k in keep}
        canon: dict[str, str] = {}
        for n in names:
            for w in _clean_words(n):
                key = _squash(w)
                if len(key) >= 3 and key not in keep_l and w.lower() not in keep_l:
                    canon.setdefault(key, w)
        self.names = canon
        self.handles = sorted({h.lower().lstrip("@") for h in handles if h and len(h.strip("@")) >= 3} - keep_l,
                              key=len, reverse=True)
        self._back: dict[str, str] = {}
        parts: list[tuple[str, str]] = []
        for h in self.handles:
            self._back[_code("handle", h).lower()] = "@" + h
            parts.append((rf"(?<![A-Za-z0-9_.])@?{re.escape(h)}(?![A-Za-z0-9_])", "h:" + h))
        for key, disp in sorted(canon.items(), key=lambda kv: len(kv[0]), reverse=True):
            self._back[_code("name", key).lower()] = disp
            parts.append((r"(?<![A-Za-z0-9])" + "".join(re.escape(c) + "+" for c in key) + r"(?![A-Za-z])", "n:" + key))
        self._tags = [t for _, t in parts]
        self._rx = re.compile("|".join(f"({p})" for p, _ in parts), re.I) if parts else None
        self._back_rx = re.compile(
            r"(?<![A-Za-z0-9])(" + "|".join(re.escape(c) for c in sorted(self._back, key=len, reverse=True)) + r")(?![A-Za-z0-9])",
            re.I) if self._back else None

    def mask(self, text: str) -> str:
        if not text or self._rx is None:
            return text

        def rep(m: re.Match[str]) -> str:
            kind, key = self._tags[(m.lastindex or 1) - 1].split(":", 1)
            return _code("handle" if kind == "h" else "name", key)

        return self._rx.sub(rep, text)

    def unmask(self, text: str) -> str:
        if not text or self._back_rx is None:
            return text
        return self._back_rx.sub(lambda m: self._back.get(m.group(1).lower(), m.group(1)), text)

    def mask_any(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.mask(obj)
        if isinstance(obj, list):
            return [self.mask_any(x) for x in obj]
        if isinstance(obj, dict):
            return {k: (v if k in _KEEP_KEYS else self.mask_any(v)) for k, v in obj.items()}
        return obj

    def unmask_any(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.unmask(obj)
        if isinstance(obj, list):
            return [self.unmask_any(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.unmask_any(v) for k, v in obj.items()}
        return obj

    def names_new_people(self, text: str, allowed: str) -> bool:
        """True if `text` names someone who isn't in `allowed` (used to stop a fine-tuned voice from bringing in
        people from other chats it memorised)."""
        ok = _squash(allowed or "")
        return any(_squash(w) in self.names and _squash(w) not in ok for w in re.findall(r"[A-Za-z]{3,}", text or ""))


# ── where the names come from ─────────────────────────────────────────────────
def _label_names(label: str) -> list[str]:
    out = []
    for part in re.split(r"\s*(?:,|/|&|\band\b)\s*", label.strip()):
        words = part.split()
        if not words:
            continue
        if len(words) <= 3 and all(w[:1].isupper() for w in words if w.isalpha()):
            out.append(part)
        elif words[0][:1].isupper() and len(words) > 1 and words[1].lower() in ("the", "sir", "bhaiya", "didi"):
            out.append(words[0])
    return out


def _common_words(chunks_path: Path, top: int = 600) -> set[str]:
    c: Counter[str] = Counter()
    try:
        with chunks_path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i > 6000:
                    break
                c.update(_squash(w) for w in re.findall(r"[a-z]{2,}", json.loads(line).get("text", "").lower()))
    except (OSError, ValueError):
        return set()
    return {w for w, _ in c.most_common(top)}


def _lines(path: Path) -> list[str]:
    try:
        return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    except OSError:
        return []


def collect_names(me_dir: Path, data_dir: Path, profile: dict[str, Any] | None = None,
                  db_people: Iterable[tuple[str, str]] = ()) -> tuple[list[str], list[str], list[str]]:
    """(names, handles, keep) from your data; see the module docstring."""
    sure: list[str] = []
    guessed: list[str] = []
    handles: list[str] = []
    chunks = data_dir / "chats.jsonl"
    if chunks.exists():
        seen: set[str] = set()
        with chunks.open(encoding="utf-8") as fh:
            for line in fh:
                for n in json.loads(line).get("names") or []:
                    if n not in seen:
                        seen.add(n)
                        sure.append(n)
    for f in sorted((me_dir / "people").glob("*.md")):
        text = f.read_text(encoding="utf-8", errors="replace")
        for t in _TITLE.findall(text)[:1]:
            guessed.append(t)
            handles += re.findall(r"@([A-Za-z0-9_.]{3,30})", t)
        for block in _WORLD.findall(text):
            for label in _BOLD_LABEL.findall(block):
                guessed += _label_names(label)
    p = profile or {}
    for key in ("name", "nickname"):
        if p.get(key):
            sure.append(str(p[key]))
    for person in (p.get("family") or []) + (p.get("people") or []):
        if isinstance(person, dict) and person.get("name"):
            sure.append(str(person["name"]))
    for handle, name in db_people:
        if handle:
            handles.append(handle)
        if name:
            sure.append(name)
    sure += _lines(me_dir / "mask_extra.txt")
    common = _common_words(chunks)
    names = sure + [w for n in guessed for w in _clean_words(n)
                    if len(w) >= 4 and not all(_squash(x) in common for x in _clean_words(n))]
    sure_keys = {_squash(w) for n in sure for w in _clean_words(n)}
    keep = _lines(me_dir / "mask_keep.txt") + sorted(_ENGLISH | (common - sure_keys))
    return names, handles, keep


_lock = threading.Lock()
_cached: tuple[float, NameMasker | None] = (0.0, None)


def get_masker(force: bool = False) -> NameMasker | None:
    """The masker for this process (None when MASK_NAMES is off). Rebuilt every 30 minutes."""
    global _cached
    from replace_me.config import get_settings

    s = get_settings()
    if not s.MASK_NAMES:
        return None
    with _lock:
        at, m = _cached
        if m is not None and not force and time.monotonic() - at < 1800:
            return m
        try:
            from replace_me.profile import load_profile
            from replace_me.store import Store

            db_people = Store().people() if s.db_path.exists() else []
            names, handles, keep = collect_names(s.ME_DIR, s.DATA_DIR, load_profile(), db_people)
            m = NameMasker(names, handles, keep)
            logger.info("masking.ready", names=len(m.names), handles=len(m.handles))
        except Exception as e:
            logger.warning("masking.build_failed", error=str(e)[:200])
            m = None
        _cached = (time.monotonic(), m)
        return m
