"""The inbox: one tab reads the chat list every few seconds; each chat with something new gets its own tab and task,
so several people are answered at once (the slow part, the model writing, overlaps). After a reply the chat stays
open and is re-read every INBOX_HOT_POLL_SECONDS for INBOX_HOT_SECONDS: if they write again the answer comes right
away, then the tab closes. Only one tab types at a time. Message requests are never opened.

Lessons baked in (each was a real bug):
  - muted people are re-checked before reading and right before sending, also inside the hot follow-up
  - a login/challenge page is left alone (reloading it every few seconds made logging in impossible)
  - each reel in a burst gets its own answer, quoting that reel
  - what you typed yourself (bubbles after the bot's last one that it never sent) is learnt
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from replace_me.config import get_settings
from replace_me.instagram import dom
from replace_me.instagram.chat import Bubble, ChatTab, SecurityCheck
from replace_me.log import get_logger
from replace_me.replier import Replier

logger = get_logger("instagram.inbox")
REEL = "[Shared a Reel]"
_THEIRS = re.compile(r"^\s*(?:you\b|you:|sent\b|seen\b|delivered\b|active\b)", re.I)
_RECENT = re.compile(r"^\s*(?:now|just now|(\d+)\s*[ms])\s*$", re.I)


@dataclass
class Row:
    name: str
    preview: str
    time: str
    bold: bool

    @property
    def from_them(self) -> bool:
        return self.bold or (bool(self.preview) and not _THEIRS.match(self.preview))

    @property
    def recent(self) -> bool:
        m = _RECENT.match(self.time or "")
        return bool(m) and (m.group(1) is None or int(m.group(1)) <= 30 or self.time.strip().lower().endswith("s"))


def reel_segments(texts: list[str]) -> list[list[int]]:
    """Each reel answered on its own with the texts right after it (usually about it); no reel = one reply."""
    groups: list[list[int]] = []
    for i, t in enumerate(texts):
        if REEL in t or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
    if not any(REEL in texts[g[0]] for g in groups):
        return [list(range(len(texts)))] if texts else []
    return groups


class Inbox:
    def __init__(self, context: Any, replier: Replier | None = None):
        s = get_settings()
        self.context = context
        self.replier = replier or Replier()
        self.store = self.replier.store
        self.page: Any = None
        self.seen: dict[str, str] = {}
        self.latest: dict[str, str] = {}
        self.threads: dict[str, str] = {}
        self.tabs: dict[str, ChatTab] = {}
        self.active: dict[str, asyncio.Task] = {}
        self.typing = asyncio.Lock()
        self.fatal: BaseException | None = None
        self.running = False
        self.max_parallel = max(1, s.INBOX_MAX_PARALLEL)

    async def _new_tab(self) -> Any:
        page = await self.context.new_page()
        await page.set_viewport_size(dom.VIEWPORT)
        return page

    async def _goto(self, page: Any, url: str) -> None:
        await page.goto(url, wait_until="domcontentloaded", timeout=get_settings().NAVIGATION_TIMEOUT_MS)
        await asyncio.sleep(random.uniform(3.0, 4.5))

    async def _rows(self) -> list[Row]:
        try:
            return [Row(**r) for r in await self.page.evaluate(dom.INBOX_ROWS)]
        except Exception:
            return []

    def _muted(self, handle: str = "", name: str = "") -> bool:
        return self.store.is_muted(handle=handle, name=name)

    async def run(self) -> None:
        s = get_settings()
        self.running = True
        self.page = await self._new_tab()
        await self._goto(self.page, dom.INBOX_URL)
        logger.info("inbox.started", parallel=self.max_parallel, hot_seconds=s.INBOX_HOT_SECONDS)
        first, failures, waiting_login, reloaded = True, 0, False, time.monotonic()
        try:
            while self.running:
                if self.fatal is not None:
                    raise self.fatal
                try:
                    if re.search(dom.LOGIN_PAGE, self.page.url or ""):
                        if not waiting_login:
                            logger.warning("inbox.waiting_for_login", hint="log in in the browser window; the bot waits")
                        waiting_login = True
                        await asyncio.sleep(10)
                        continue
                    waiting_login = False
                    if time.monotonic() - reloaded > 180 or not self.page.url.startswith(dom.INBOX_URL):
                        await self._goto(self.page, dom.INBOX_URL)
                        reloaded = time.monotonic()
                    await self._scan(first)
                    first, failures = False, 0
                except (SecurityCheck, asyncio.CancelledError):
                    raise
                except Exception as e:
                    failures += 1
                    wait = min(300.0, s.INBOX_SCAN_SECONDS * 2 ** min(failures, 4))
                    logger.warning("inbox.scan_failed", error=str(e)[:200], retry_in=round(wait))
                    await asyncio.sleep(wait)
                    continue
                await asyncio.sleep(s.INBOX_SCAN_SECONDS + random.uniform(0, 1.5))
        finally:
            self.running = False
            for t in list(self.active.values()):
                t.cancel()

    async def _scan(self, first: bool) -> None:
        rows = await self._rows()
        todo = []
        for r in rows:
            if not r.name or r.name in self.active or not r.from_them or self._muted(name=r.name):
                continue
            if (first and (r.recent or r.bold)) or (not first and self.seen.get(r.name) != r.preview):
                todo.append(r)
        for r in rows:
            self.latest[r.name] = r.preview
            if r.name not in self.active:
                self.seen[r.name] = r.preview
        for row in todo:
            if len(self.active) >= self.max_parallel:
                self.seen.pop(row.name, None)  # all tabs busy: next scan
                continue
            self.active[row.name] = asyncio.create_task(self._chat(row))

    async def _open(self, page: Any, name: str) -> str | None:
        url = self.threads.get(name)
        if url:
            await page.goto(url, wait_until="domcontentloaded", timeout=get_settings().NAVIGATION_TIMEOUT_MS)
        else:
            await self._goto(page, dom.INBOX_URL)
            if not await page.evaluate(dom.CLICK_ROW, name):
                return None
        for _ in range(20):
            await asyncio.sleep(0.5)
            if "/direct/t/" in page.url:
                break
        else:
            return None
        for _ in range(12):
            await asyncio.sleep(0.5)
            user = (await page.evaluate(dom.HEADER_USER) or "").strip().lower()
            if user:
                self.threads[name] = page.url.split("?")[0]
                return user
        return None

    async def _chat(self, row: Row) -> None:
        s = get_settings()
        page, snap = None, self.latest.get(row.name, row.preview)
        try:
            page = await self._new_tab()
            handle = await self._open(page, row.name)
            if not handle or self._muted(handle, row.name):
                return
            tab = self.tabs.setdefault(handle, ChatTab(page))
            tab.page = page
            if not await self._answer(tab, handle, row.name):
                return
            deadline = time.monotonic() + s.INBOX_HOT_SECONDS
            while self.running and time.monotonic() < deadline:
                await asyncio.sleep(s.INBOX_HOT_POLL_SECONDS)
                if self._muted(handle, row.name):
                    break
                snap = self.latest.get(row.name, snap)
                if await self._answer(tab, handle, row.name):
                    deadline = time.monotonic() + s.INBOX_HOT_SECONDS  # they replied: another window
        except SecurityCheck as e:
            self.fatal = e
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("inbox.chat_failed", error=str(e)[:200])
            snap = None
        finally:
            self.active.pop(row.name, None)
            if snap is None:
                self.seen.pop(row.name, None)
            else:
                self.seen[row.name] = snap
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass

    async def _answer(self, tab: ChatTab, handle: str, name: str) -> bool:
        """Read the open chat, answer anything new. True when something was sent."""
        if self._muted(handle, name):
            return False
        await tab.check()
        visible = await tab.read()
        self.store.learn_manual(handle, [b.text for b in visible if b.mine])
        pending = tab.new_messages(visible)
        if pending:
            pending, visible = await tab.settle(pending, visible)
        tab.took_in(visible)
        if not pending:
            return False
        texts = [b.text for b in pending]
        groups = reel_segments(texts)
        sent = False
        for g in groups:
            if self._muted(handle, name):
                return sent
            reply = await self.replier.answer(handle, name, [texts[i] for i in g])
            if not reply.bubbles or self._muted(handle, name):  # switched off while writing: never send
                if reply.bubbles:
                    self.store.drop_last_mine(handle)
                continue
            await asyncio.sleep(random.uniform(1.0, 3.0) if get_settings().HUMAN_TIMING else 0.2)
            async with self.typing:
                try:
                    await tab.page.bring_to_front()
                except Exception:
                    pass
                first = pending[g[0]]
                if len(groups) > 1 and REEL in first.text and first.ref:
                    await tab.quote(ref=first.ref)
                elif reply.reply_to:
                    await tab.quote(text=reply.reply_to)
                try:
                    await tab.send(reply.bubbles)
                except Exception:
                    self.store.drop_last_mine(handle)
                    raise
            sent = True
            logger.info("inbox.replied", to=handle, bubbles=len(reply.bubbles), ms=round(reply.latency_ms))
        tab.took_in(await tab.read())
        return sent
