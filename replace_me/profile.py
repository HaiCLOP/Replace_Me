"""me/profile.yaml (who you are, the people in your life) and the text files next to it."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from replace_me.config import get_settings


def load_profile(path: Path | None = None) -> dict[str, Any]:
    p = path or (get_settings().ME_DIR / "profile.yaml")
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def read_note(path: Path) -> str:
    """A markdown note without HTML comments and '#'-only comment lines meant for the reader."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    return re.sub(r"<!--.*?-->\s*", "", text, flags=re.S).strip()


def life_now() -> str:
    """me/now.md: what's going on in your life right now (lines starting with # are ignored)."""
    text = read_note(get_settings().ME_DIR / "now.md")
    return "\n".join(l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")).strip()


def master_profile() -> str:
    return read_note(get_settings().people_dir / "_me.md")


def person_note(chats: list[str]) -> str:
    """Your notes on the person behind these export chats (tools/research_people.py), first one found.
    The 'Real exchanges' section and lists of typical lines are left out: a model pastes them word for word."""
    for c in chats:
        text = read_note(get_settings().people_dir / f"{c}.md")
        if text:
            text = re.split(r"(?m)^##\s+Real exchanges.*$", text)[0]
            out, skipping = [], False
            for line in text.splitlines():
                if re.match(r"^\s*Typical lines", line, re.I):
                    skipping = True
                    continue
                if skipping and (re.match(r"^\s*[-*]\s", line) or not line.strip()):
                    continue
                skipping = False
                out.append(line)
            return "\n".join(out).strip()
    return ""


def person(profile: dict[str, Any], handle: str) -> dict[str, Any]:
    """The profile's entry for an Instagram handle (score, clean, label), or {}."""
    h = (handle or "").lower().lstrip("@")
    for p in profile.get("people") or []:
        if isinstance(p, dict) and str(p.get("handle", "")).lower().lstrip("@") == h:
            return p
    return {}


def is_bot_aware(profile: dict[str, Any], handle: str) -> bool:
    return (handle or "").lower().lstrip("@") in {str(h).lower().lstrip("@") for h in profile.get("bot_aware_handles") or []}
