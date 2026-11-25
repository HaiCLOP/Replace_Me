"""One reply, start to finish. Called by replier.py with everything it gathered about this chat."""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from replace_me.brain import filters, prompt, request, select
from replace_me.brain.lookups import STORY_ASK, Lookups
from replace_me.config import get_settings
from replace_me.llm.chain import build_chain
from replace_me.log import get_logger
from replace_me.memory.corpus import DEFAULT_SWEARS, Corpus

logger = get_logger("brain.reply")
_BOT_Q = re.compile(r"\b(?:are you|r u|is this|tum|tu|aap)\b.{0,20}\b(?:a\s+)?(?:bot|ai|robot|chatgpt|auto.?reply)\b", re.I)


@dataclass
class Situation:
    handle: str
    label: str                      # "Riya (@riya.k)" as the copy should think of them
    incoming: list[str]             # their new message(s), oldest first
    history: list[Any]              # store.Msg window for the cached turns
    person_note: str = ""
    latest: list[str] = field(default_factory=list)       # your latest real lines with them (export)
    my_days: list[str] = field(default_factory=list)      # what you typed lately in other chats
    close: bool = False             # score >= GOSSIP_MIN_SCORE: may hear about other chats
    clean: bool = False             # family/respect: no swearing
    lookups: Lookups = field(default_factory=Lookups)
    known_names: list[str] = field(default_factory=list)
    gender: str = ""                # optional hint, helps when names are masked
    bot_aware: bool = False         # knows it's a bot (your alt): skip the honesty rule


@dataclass
class Reply:
    bubbles: list[str]
    reply_to: str = ""
    searches: list[str] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    cache: dict[str, int] = field(default_factory=dict)
    told_gossip: bool = False
    note: str = ""


async def write_reply(sit: Situation, profile: dict[str, Any], corpus: Corpus, master: str, life_now: str,
                      seed: int | None = None) -> Reply:
    s = get_settings()
    rng = random.Random(seed)
    t0 = time.perf_counter()
    stats = corpus.load().stats
    swears = str(profile.get("swear_words") or DEFAULT_SWEARS)
    incoming = [m for m in sit.incoming if m and m.strip()]
    story = bool(STORY_ASK.search(" ".join(incoming[-2:])))

    # 1. what's looked up for sure + what this turn adds
    extras: list[str] = []
    found = sit.lookups.always(incoming, sit.known_names)
    if found:
        extras.append("WHAT YOU REMEMBER (real, from your history; use it if it answers them, otherwise ignore it; "
                      "never say you looked it up):\n" + "\n\n".join(found))
    if sit.my_days:
        extras.append("WHAT YOU'VE BEEN UP TO THESE DAYS (your own real messages in other chats: you can bring one up or "
                      "jump to it like you do; never force it, never copy a line):\n" + "\n".join(f"- {t}" for t in sit.my_days))
    if story and sit.close:
        if sit.lookups.gossip:
            extras.append("WHAT'S BEEN GOING ON IN YOUR OTHER CHATS (they're one of your closest, you can tell them):\n"
                          + "\n".join(f"- {g}" for g in sit.lookups.gossip))
        extras.append("If they want the story / the tea and you know it: tell it like you'd tell a friend (who, what "
                      "happened, the juicy bit, your take), the main part now, more if they ask. Don't dodge when you "
                      "know it; never invent a detail; if you don't know a part, say so.")
    elif story:
        extras.append("This person is NOT one of your closest: don't tell them other people's stuff or gossip (dodge it "
                      "your way). Things about the two of you are fine.")

    n = filters.pick_burst(stats.get("burst_sizes", {}), rng)
    if story and (found or sit.close):
        n = max(n, rng.choice([2, 3, 3, 4]))
    k = s.CANDIDATES
    examples = corpus.find(" ".join(incoming[-3:]), k=16, avoid=[m.text for m in sit.history[-8:] if m.who == "me"], seed=seed)
    bot_q = not sit.bot_aware and any(_BOT_Q.search(m) for m in incoming)
    label = f"{sit.label} (a {sit.gender})" if sit.gender in ("girl", "boy") else sit.label

    chain = build_chain(s.LLM_CHAIN)
    use_tools = s.USE_TOOLS and chain.can_use_tools()
    system = prompt.build_system(profile, stats, master, life_now, tools=use_tools)
    person = prompt.build_person_block(label, sit.person_note, profile.get("codenames") or {}, sit.latest, sit.clean)
    now = prompt.build_now_block(examples, incoming, extras, n, several=sum(not m.startswith("[") for m in incoming) >= 2,
                                 bot_question=bot_q)
    if k > 1:
        now += prompt.CANDIDATES_ASK.format(k=k)
    max_tokens = 350 + 120 * k if k > 1 else None

    # 2. write: cached request with tools, or one plain prompt as the fallback
    res = None
    if use_tools:
        try:
            sys_blocks, msgs = request.build(system, person, request.as_turns(sit.history, incoming), now)
            res = await chain.generate_with_tools(sys_blocks, msgs, sit.lookups.tools(), sit.lookups.run, 2, max_tokens)
        except Exception as e:
            logger.warning("reply.tools_failed", error=str(e)[:200])
    if res is None:
        transcript = "\n".join((f"Them: {m.text}" if m.who == "them" else f"You: {m.text}") for m in sit.history[-24:])
        res = await chain.generate(person + "\n" + transcript + "\n\n" + now, system, max_tokens=max_tokens)

    # 3. clean every candidate
    theirs = incoming + [m.text for m in sit.history[-8:] if m.who == "them"]
    raw, reply_to = select.parse_reply_to(res.content or "", theirs)
    your_rate = float(stats.get("swear_rate", 0.02))

    def finish(text: str) -> list[str]:
        lines = filters.clean_raw(text)
        lines = filters.strip_swears(lines, swears) if sit.clean else filters.thin_swears(lines, your_rate, rng, swears)
        lines = filters.drop_money_asks(filters.drop_echoes(lines, incoming))
        return filters.fit(lines, n)

    options: list[list[str]] = []
    for part in select.split_candidates(raw) if k > 1 else [raw]:
        o = finish(part)
        if o and o not in options:
            options.append(o)

    # 4. option C: your fine-tuned model words every candidate, the picker judges the words that get sent
    tail = [("Them: " if m.who == "them" else "You: ") + m.text.replace("\n", " / ") for m in sit.history[-24:]]
    who = sit.label.split(" (")[0]
    if s.VOICE in ("meaning_all", "meaning") and options:
        targets = options if s.VOICE == "meaning_all" else options[:1]
        worded = []
        for o in targets:
            mine = finish("\n".join(await select.in_your_voice(o, tail, incoming, profile.get("name", ""), who)))
            if mine and mine not in worded:
                worded.append(mine)
        if worded:
            options = worded if s.VOICE == "meaning_all" or len(options) == 1 else worded + options[1:]

    # 5. pick
    chosen = options[0] if options else []
    if len(options) > 1 and s.PICKER == "judge":
        fit = filters.by_your_length(options, stats.get("length_mix", []), rng)
        chosen = fit[await select.pick_like_a_judge(fit, tail, incoming, corpus.my_lines(), rng)] if len(fit) > 1 else fit[0]

    told = bool(sit.close and sit.lookups.gossip and story)
    logger.info("reply.done", to=sit.handle, out=" / ".join(chosen)[:120], lookups=sit.lookups.log, options=len(options),
                model=res.model, cache=res.cache)
    return Reply(bubbles=chosen, reply_to=reply_to, searches=list(sit.lookups.log), model=res.model,
                 latency_ms=(time.perf_counter() - t0) * 1000, cache=res.cache, told_gossip=told)
