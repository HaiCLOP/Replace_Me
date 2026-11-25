"""One open chat in one browser tab: read it, know what's new, type a reply like a person, quote a message."""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from replace_me.config import get_settings
from replace_me.instagram import dom
from replace_me.log import get_logger

logger = get_logger("instagram.chat")


class SecurityCheck(Exception):
    """Instagram showed a security check / block. Everything stops; a human has to look."""


@dataclass
class Bubble:
    text: str
    mine: bool
    ref: str | None = None


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", (s or "").lower()).split())


class ChatTab:
    def __init__(self, page: Any):
        self.page = page
        self.snapshot: list[tuple[bool, str]] = []  # what was on screen when we last took messages in
        self.sent: list[tuple[float, str]] = []     # what we typed (its echo on screen is ours, not theirs)

    # ── reading ──────────────────────────────────────────────────────────────
    async def read(self) -> list[Bubble]:
        try:
            data = await self.page.evaluate(dom.MESSAGES) or []
        except Exception as e:
            logger.debug("chat.read_failed", error=str(e)[:160])
            return []
        out = []
        for d in data:
            text = str(d.get("text") or "").strip()
            if text:
                out.append(Bubble(text=text, mine=bool(d.get("mine")) or self._is_echo(text), ref=d.get("ref")))
        return out

    def _is_echo(self, text: str) -> bool:
        now = time.time()
        self.sent = [(t, s) for t, s in self.sent if now - t < 180]
        n = _norm(text)
        return bool(n) and any(n == s or (len(n) > 4 and (n in s or s in n)) for _, s in self.sent)

    @staticmethod
    def _seq(bubbles: Sequence[Bubble]) -> list[tuple[bool, str]]:
        return [(b.mine, _norm(b.text)) for b in bubbles]

    def new_messages(self, bubbles: Sequence[Bubble]) -> list[Bubble]:
        """Their messages after the point we last took messages in (anchored on the last few we saw, longest
        match first), or on a first look: their unanswered tail."""
        seq, snap = self._seq(bubbles), self.snapshot
        for n in range(min(6, len(snap)), 0, -1):
            tail = snap[-n:]
            for end in range(len(seq) - 1, n - 2, -1):
                if seq[end - n + 1:end + 1] == tail:
                    return [b for b in bubbles[end + 1:] if not b.mine]
        trailing: list[Bubble] = []
        for b in reversed(bubbles):
            if b.mine:
                break
            trailing.append(b)
        return list(reversed(trailing))

    def took_in(self, bubbles: Sequence[Bubble]) -> None:
        self.snapshot = self._seq(bubbles)[-30:]

    async def settle(self, pending: list[Bubble], visible: list[Bubble]) -> tuple[list[Bubble], list[Bubble]]:
        """Wait while they're still sending (a burst), up to a few rounds."""
        wait = get_settings().BURST_SETTLE_SECONDS
        for _ in range(4):
            if wait <= 0:
                break
            await asyncio.sleep(wait)
            visible = await self.read()
            fresh = self.new_messages(visible)
            if [b.text for b in fresh] == [b.text for b in pending]:
                break
            pending = fresh
        return pending, visible

    # ── safety ───────────────────────────────────────────────────────────────
    async def check(self) -> None:
        if re.search(dom.LOGIN_PAGE, self.page.url or ""):
            raise SecurityCheck(f"logged out / challenge page: {self.page.url}")
        for sel in dom.CHECKPOINTS:
            try:
                el = await self.page.query_selector(sel)
                if el and await el.is_visible():
                    raise SecurityCheck(f"Instagram security check ({sel}) at {self.page.url}")
            except SecurityCheck:
                raise
            except Exception:
                continue

    async def message_box(self) -> Any:
        for sel in dom.MESSAGE_BOX:
            try:
                el = await self.page.query_selector(sel)
                if el and await el.is_visible():
                    return el
            except Exception:
                continue
        return None

    # ── writing ──────────────────────────────────────────────────────────────
    async def quote(self, text: str = "", ref: str | None = None) -> bool:
        """Instagram's swipe-reply on a message (by its text) or a reel card (by ref): what's typed next quotes it."""
        pos = await self.page.evaluate(dom.FIND_BUBBLE, [text, ref])
        if not pos:
            return False
        await self.page.mouse.move(pos["x"], pos["y"])
        await asyncio.sleep(random.uniform(0.5, 0.9))
        btn = await self.page.evaluate(dom.HOVER_BUTTON, ["^reply|reply to", pos["y"], max(70, (pos.get("h") or 0) / 2 + 40)])
        if not btn:
            return False
        await self.page.mouse.click(btn["x"], btn["y"])
        await asyncio.sleep(random.uniform(0.4, 0.8))
        logger.info("chat.quoting", via=btn.get("label", ""))
        return True

    async def send(self, bubbles: Sequence[str]) -> None:
        human = get_settings().HUMAN_TIMING
        for i, bubble in enumerate(b.strip() for b in bubbles if b.strip()):
            await self.check()
            box = await self.message_box()  # found again every time: sending (or a quote bar closing) re-renders it
            if box is None:
                raise RuntimeError("message box not found")
            await box.click()
            await asyncio.sleep(random.uniform(0.15, 0.3))
            for ch in bubble:
                await self.page.keyboard.type(ch, delay=random.uniform(55, 105) if human else random.uniform(15, 35))
                if human and ch in ".,!? " and random.random() < 0.08:
                    await asyncio.sleep(random.uniform(0.2, 0.6))
            await asyncio.sleep(random.uniform(0.12, 0.25))
            await self.page.keyboard.press("Enter")
            self.sent.append((time.time(), _norm(bubble)))
            if i < len(bubbles) - 1:
                await asyncio.sleep(random.uniform(0.5, 1.6) if human else 0.4)
