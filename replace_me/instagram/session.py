"""A persistent Chromium profile (browser_profile/, git-ignored) so you log in to Instagram once, by hand."""

from __future__ import annotations

from typing import Any

from replace_me.config import get_settings

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36")


async def launch(headless: bool = False) -> tuple[Any, Any]:
    from playwright.async_api import async_playwright

    s = get_settings()
    pw = await async_playwright().start()
    ctx = await pw.chromium.launch_persistent_context(
        user_data_dir=str(s.BROWSER_PROFILE), headless=headless, viewport={"width": 1280, "height": 900},
        user_agent=USER_AGENT, args=["--disable-blink-features=AutomationControlled"],
    )
    return pw, ctx


async def login() -> None:
    """Open a browser on Instagram's login page; you log in (and do any 2FA) yourself, then press Enter here."""
    pw, ctx = await launch(headless=False)
    page = ctx.pages[0] if ctx.pages else await ctx.new_page()
    await page.goto("https://www.instagram.com/accounts/login/")
    print("\nLog in to Instagram in the browser window (2FA too). When your inbox is visible, press Enter here.")
    input()
    await ctx.close()
    await pw.stop()
    print("Saved. `python -m replace_me run` will use this login.")
