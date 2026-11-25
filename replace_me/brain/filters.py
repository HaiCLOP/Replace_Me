"""What happens to the model's text before it's sent. Each rule came from a blind-judge tell (docs/EVALUATION.md)."""

from __future__ import annotations

import random
import re
from typing import Any, Sequence

_META = re.compile(r"^\s*(?:thought|thinking|note|reply|search)\s*[:\-]", re.I)
_PREFIX = re.compile(r"^\s*(?:you|me)\s*:\s*", re.I)
_NARRATION = re.compile(r"\b(?:from|in|based on|according to)\s+(?:the|our|my|your)\s+(?:earlier|past|previous|old)\s+"
                        r"(?:chat|chats|messages?|history)|\bchat history\b|\blooked (?:it )?up\b", re.I)
_MONEY = re.compile(r"\b(?:send|bhej\w*|transfer|pay|upi|gpay|paytm|paise|paisa|money|rupees?|rs\.?|otp|password)\b", re.I)
_ASK = re.compile(r"\?|\b(?:kya|kyu|kaise|kab|kaha|kaun|what|why|how|when|where|who)\b", re.I)


def clean_raw(text: str, max_lines: int = 6) -> list[str]:
    """The model's output -> message lines: no labels, no 'Them:' continuation, no 'I looked it up'."""
    out: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or _META.match(s) or s in ("---", "```", "==="):
            continue
        if s.lower().startswith("them:"):
            break
        if _NARRATION.search(s):
            continue
        s = _PREFIX.sub("", s).strip().strip('"')
        if s:
            out.append(s)
    return out[:max_lines]


def strip_swears(lines: list[str], pattern: str) -> list[str]:
    rx = re.compile(pattern, re.I)
    out = []
    for line in lines:
        t = re.sub(r"\s{2,}", " ", rx.sub(" ", line)).strip(" ,")
        if re.search(r"\w", t):
            out.append(t)
    return out


def thin_swears(lines: list[str], your_rate: float, rng: random.Random, pattern: str) -> list[str]:
    """Swear at YOUR rate: models swear in 10-17% of replies when told 'be raw'; a real person may be at 1-2%."""
    rx = re.compile(pattern, re.I)
    if not any(rx.search(l) for l in lines):
        return lines
    keep = min(1.0, max(0.02, your_rate * 8))  # some headroom: a reply has a few lines
    return lines if rng.random() < keep else strip_swears(lines, pattern)


_COMMON = {"the", "and", "you", "that", "this", "what", "have", "with", "your", "haa", "nahi", "kya", "hai", "bhai",
           "yaar", "acha", "toh", "bhi", "kar", "raha", "rahi", "kuch", "abhi", "aaj", "kal"}


def drop_echoes(lines: list[str], incoming: Sequence[str]) -> list[str]:
    """A line that is mostly their own words handed back ('good night sweet dreams' -> the same) goes, if there's
    another line: the first thing a blind judge spots."""
    theirs = {re.sub(r"(.)\1+", r"\1", w) for m in incoming[-3:] for w in re.findall(r"[a-z]{3,}", (m or "").lower())} - _COMMON
    if not theirs:
        return lines
    keep = []
    for line in lines:
        own = [w for w in (re.sub(r"(.)\1+", r"\1", x) for x in re.findall(r"[a-z]{3,}", line.lower())) if w not in _COMMON]
        echoed = sum(w in theirs for w in own)
        if own and echoed >= 2 and echoed / len(own) >= 0.6:
            continue
        keep.append(line)
    return keep or lines


def drop_money_asks(lines: list[str]) -> list[str]:
    """Never ask anyone for money, an OTP or a password, not even as a joke."""
    out = [l for l in lines if not (_MONEY.search(l) and _ASK.search(l))]
    return out if (out or not lines) else ["ok"]


def fit(lines: list[str], n: int) -> list[str]:
    """At most n messages and at most one question (people rarely ask two things at once)."""
    out, asked = [], False
    for line in lines:
        q = "?" in line
        if q and asked:
            continue
        asked = asked or q
        out.append(line)
        if len(out) >= n:
            break
    return out


def length_bucket(words: int) -> int:
    return 0 if words <= 1 else 1 if words <= 5 else 2 if words <= 12 else 3


def by_your_length(options: list[list[str]], length_mix: Sequence[float], rng: random.Random) -> list[list[str]]:
    """Options whose length matches one drawn from YOUR real mix of reply lengths. Left free, a judge-style
    picker drifts to one-word replies (83-90% vs a real 63%): they fool a judge but make the copy dull."""
    mix = list(length_mix) if length_mix and sum(length_mix) > 0 else [0.2, 0.36, 0.29, 0.15]
    want = rng.choices(range(4), weights=mix)[0]
    buckets = [length_bucket(sum(len(l.split()) for l in o)) for o in options]
    for d in range(4):  # nearest bucket that has options
        fit_ = [o for o, b in zip(options, buckets) if abs(b - want) == d]
        if fit_:
            return fit_
    return options


def pick_burst(burst_sizes: dict[str, float], rng: random.Random) -> int:
    sizes = {int(k): v for k, v in (burst_sizes or {}).items()} or {1: 0.6, 2: 0.26, 3: 0.08, 4: 0.04, 5: 0.02}
    return rng.choices(list(sizes), weights=list(sizes.values()))[0]
