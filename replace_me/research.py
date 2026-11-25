"""Notes on each person, and your master profile, written by a strong model that reads your real chats once.

This is what turns "an AI that texts" into "an AI that texts like YOU with THIS person": the per-person notes were
the biggest single quality jump in testing. A model reads each chat (recent days in full, older days sampled for
huge chats) next to hard numbers counted locally, and has to back every claim with your real lines.

    python -m replace_me research people [--only <chat_folder> ...] [--min 40] [--force]
    python -m replace_me research me        # master profile from all the notes

Writes me/people/<chat_folder>.md and me/people/_me.md (git-ignored). Names are masked on the way to the model
when MASK_NAMES is on, and unmasked in the notes it writes. Cost: roughly one long model call per chat.
"""

from __future__ import annotations

import asyncio
import collections
import json
import math
import re
import statistics
from datetime import datetime, timezone
from typing import Any

from replace_me.config import get_settings
from replace_me.llm.chain import build_chain
from replace_me.memory.corpus import DEFAULT_SWEARS
from replace_me.memory.export import Chat, read_chats
from replace_me.profile import load_profile

_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
TRANSCRIPT_CHARS = 450_000

PERSON_SYSTEM = """You are a behavioural analyst building a dossier on how one specific person, {who}, texts with one specific contact. The dossier will be given to an AI that replies to this contact AS {name}, so it must capture exactly how {name} is with THIS person: not texting in general, and not a flattering portrait.

Read the whole transcript like an investigator: notice the small things people don't notice about themselves (when they go quiet, what makes them type long, what they ignore, how they change over time, what they never say to this person). Every claim must be backed by real lines, quoted exactly as typed (keep the spelling). If you're inferring, say "(guess)". Never invent events or facts. "ME:" lines are {name}.

Write in English, in this exact structure, about 1200-1800 words:

# <their name / handle as it appears>
## Who they are to me
(relationship, how close, how it started and changed over time with dates, where it stands now, who carries the chat)
## How I talk to them
(language mix, length, energy, swearing, emoji, what I call them and what they call me, how formal or rough; 8-12 of my real lines most typical WITH THEM)
## Their world
(people we both talk about and who each one is, places, school/work stuff, codenames, recurring topics, in-jokes: each with a real line; list people as "- **Name:** who they are")
## What I like about them, and what gets on my nerves
## My small habits with them
(how I open and end chats, double texting, leaving on seen, reacting to their reels, dodging, caring, apologising, teasing, compliments)
## What's going on lately
(the last 1-2 months: open threads, plans, latest news, with dates)
## Rules for the AI replying as me to this person
(10-15 sharp DO / DON'T bullets specific to this person, including things that would instantly give an AI away)
## Real exchanges
(6-8 short real exchanges, them -> me, copied exactly)
"""

ME_SYSTEM = """You are a behavioural analyst writing the master profile of how {who} texts and behaves across ALL their relationships. You get one dossier per contact (written from real chats) and overall numbers. The profile will be given to an AI that texts AS them, especially for situations no dossier covers.

Look for the deep patterns a close friend would notice, not the obvious ones: defaults, how they adapt to different people (and what never changes), tells, contradictions, what they avoid, what makes them engage or go cold, humour mechanics, conflict, affection, being ignored, gossip, family, elders, strangers, group chats. Back claims with quotes from the dossiers. Mark guesses as (guess). Don't flatter, don't psychoanalyse beyond the evidence.

Write in English, this structure, about 1500-2200 words:
# How I text (master profile)
## Core personality in chats
## How I adapt to different people (a table: type of person -> how I am)
## Humour: what I find funny and how I joke
## Conversation mechanics (openers, closers, pacing, double texts, seen, topic changes, questions, length)
## Emotions: care, hurt, anger, affection, compliments, apologies
## Things I never do (that an AI would do)
## When something new happens (no example to copy): how I'd most likely react, as rules
## Instant AI giveaways to avoid
"""


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z]+", re.sub(r"(.)\1+", r"\1", s.lower()))


def chat_stats(chat: Chat, global_counts: collections.Counter, global_total: int, swears: str) -> dict[str, Any]:
    mine = [m for m in chat.messages if m.mine and not m.text.startswith("[")]
    st: dict[str, Any] = {"folder": chat.folder, "with": chat.names, "messages_total": len(chat.messages),
                          "my_messages": len(mine)}
    if not chat.messages:
        return st
    day = lambda m: datetime.fromtimestamp(m.ts / 1000, tz=timezone.utc)
    st["first"], st["last"] = day(chat.messages[0]).date().isoformat(), day(chat.messages[-1]).date().isoformat()
    st["my_share"] = round(len(mine) / len(chat.messages), 2)
    if mine:
        texts = [m.text for m in mine]
        st["my_words_per_msg_median"] = statistics.median(len(t.split()) for t in texts)
        st["my_emoji_pct"] = round(100 * sum(bool(_EMOJI.search(t)) for t in texts) / len(texts), 1)
        st["my_top_emoji"] = collections.Counter(e for t in texts for e in _EMOJI.findall(t)).most_common(6)
        st["my_swear_pct"] = round(100 * sum(bool(re.search(swears, t, re.I)) for t in texts) / len(texts), 1)
        st["my_question_pct"] = round(100 * sum("?" in t for t in texts) / len(texts), 1)
        local = collections.Counter(w for t in texts for w in _words(t))
        lt = sum(local.values()) or 1
        scored = [(math.log(((c + .5) / (lt + 1)) / ((global_counts.get(w, 0) + .5) / (global_total + 1))) * math.log(1 + c), w, c)
                  for w, c in local.items() if c >= 4 and len(w) >= 3]
        st["my_distinctive_words"] = [f"{w} ({c})" for _, w, c in sorted(scored, reverse=True)[:30]]
    return st


def transcript(chat: Chat, cap: int = TRANSCRIPT_CHARS) -> tuple[str, str]:
    first = {n: (n.split() or ["Them"])[0] for n in chat.names}
    days: dict[str, list[str]] = collections.OrderedDict()
    for m in chat.messages:
        d = datetime.fromtimestamp(m.ts / 1000, tz=timezone.utc)
        days.setdefault(d.strftime("%Y-%m-%d %a"), []).append(f"{d:%H:%M} {'ME' if m.mine else first.get(m.sender, 'Them')}: {m.text[:400]}")
    items = list(days.items())
    size = lambda ds: sum(len(k) + sum(len(x) + 1 for x in v) for k, v in ds)
    note = "complete"
    if size(items) > cap:
        recent: list[tuple[str, list[str]]] = []
        while items and size(recent) < cap * 0.6:
            recent.insert(0, items.pop())
        per_day = max(1, size(items) // max(1, len(items)))
        step = max(1, int(len(items) / max(1, (cap - size(recent)) / per_day)))
        note = f"older days sampled (every {step}th day), last {len(recent)} days in full"
        items = items[::step] + [("----- (recent, complete) -----", [])] + recent
    return "\n".join(f"== {k}\n" + "\n".join(v) for k, v in items), note


def _research_chain():
    chain = build_chain(get_settings().RESEARCH_MODEL)
    for _, c in chain.links:
        if hasattr(c, "timeout"):
            c.timeout = 900.0
    return chain


async def run_people(only: list[str], min_mine: int = 40, force: bool = False) -> None:
    s = get_settings()
    profile = load_profile()
    name = profile.get("name") or "me"
    who = f"{name}" + (f" ({profile['about'][:120]})" if profile.get("about") else "")
    swears = str(profile.get("swear_words") or DEFAULT_SWEARS)
    chats = list(read_chats(s.export_dir, set(profile.get("export_names") or [name])))
    gc = collections.Counter(w for c in chats for m in c.messages if m.mine for w in _words(m.text))
    gt = sum(gc.values())
    s.people_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in chats if (not only or c.folder in only) and sum(m.mine for m in c.messages) >= min_mine]
    chain = _research_chain()
    for i, chat in enumerate(todo, 1):
        path = s.people_dir / f"{chat.folder}.md"
        if path.exists() and not force:
            print(f"[{i}/{len(todo)}] {chat.folder}: exists (--force to redo)")
            continue
        st = chat_stats(chat, gc, gt, swears)
        tr, note = transcript(chat)
        prompt = (f"CONTACT: {', '.join(chat.names)} ({'group chat' if chat.group else 'one-to-one chat'})\n\n"
                  f"HARD NUMBERS (counted from the whole chat):\n{json.dumps(st, ensure_ascii=False)}\n\n"
                  f"TRANSCRIPT ({note}; 'ME' = {name}):\n{tr}")
        print(f"[{i}/{len(todo)}] {chat.folder}: {len(tr) // 1000}k chars ({note})...", flush=True)
        text = ""
        for attempt in range(3):
            try:
                text = (await chain.generate(prompt, PERSON_SYSTEM.format(who=who, name=name), temperature=0.4,
                                             max_tokens=16000)).content
                break
            except Exception as e:
                print("   failed:", str(e)[:160])
                await asyncio.sleep(20 * (attempt + 1))
        if text.strip():
            path.write_text(f"<!-- notes for {chat.folder}, written {datetime.now():%Y-%m-%d} from "
                            f"{st['messages_total']} messages ({note}) -->\n{text.strip()}\n", encoding="utf-8")
            print(f"   -> {path.name} ({len(text.split())} words)")


async def run_me() -> None:
    s = get_settings()
    profile = load_profile()
    name = profile.get("name") or "me"
    notes = [p.read_text(encoding="utf-8") for p in sorted(s.people_dir.glob("*.md")) if not p.name.startswith("_")]
    if not notes:
        print("no notes yet: run `python -m replace_me research people` first")
        return
    stats_path = s.DATA_DIR / "stats.json"
    stats = stats_path.read_text(encoding="utf-8") if stats_path.exists() else "{}"
    prompt = "MY OVERALL NUMBERS:\n" + stats + "\n\nDOSSIERS:\n\n" + "\n\n---\n\n".join(notes)
    text = (await _research_chain().generate(prompt, ME_SYSTEM.format(who=name), temperature=0.4, max_tokens=16000)).content
    (s.people_dir / "_me.md").write_text(f"<!-- master profile, written {datetime.now():%Y-%m-%d} from {len(notes)} notes -->\n"
                                         + text.strip() + "\n", encoding="utf-8")
    print(f"-> me/people/_me.md ({len(text.split())} words)")
