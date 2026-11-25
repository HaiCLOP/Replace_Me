"""JavaScript that reads Instagram's web pages. DOM facts were checked on the live site (late 2026) and WILL drift:
when Instagram changes its markup, this is the file to fix (docs/TROUBLESHOOTING.md).
"""

INBOX_URL = "https://www.instagram.com/direct/inbox/"
VIEWPORT = {"width": 1280, "height": 900}  # narrower windows collapse the inbox list to bare avatars

# chat rows: div[role=button][tabindex=0] with an avatar img and spans: name, preview ("You: ..." when you sent last), time
INBOX_ROWS = r"""
() => [...document.querySelectorAll('div[role="button"][tabindex="0"]')].filter(el => {
    const r = el.getBoundingClientRect();
    return r.width > 150 && r.height > 40 && r.height < 110 && el.querySelector('img');
  }).map(el => {
    const spans = [...el.querySelectorAll('span')].filter(s => s.children.length === 0 && s.textContent.trim());
    const texts = spans.map(s => s.textContent.trim()).filter(t => t !== '·');
    const bold = spans.some(s => parseInt(getComputedStyle(s).fontWeight || '400', 10) >= 600);
    return {name: texts[0] || '', preview: texts[1] || '', time: texts[texts.length - 1] || '', bold};
  })
"""

CLICK_ROW = r"""
(name) => {
  const rows = [...document.querySelectorAll('div[role="button"][tabindex="0"]')].filter(el => el.querySelector('img'));
  const row = rows.find(el => {
    const first = [...el.querySelectorAll('span')].find(s => s.children.length === 0 && s.textContent.trim());
    return first && first.textContent.trim() === name;
  });
  if (!row) return false;
  row.click();
  return true;
}
"""

HEADER_USER = r"""
() => {
  const links = [...document.querySelectorAll('a[href]')].filter(a => {
    const r = a.getBoundingClientRect(); const h = a.getAttribute('href') || '';
    return r.top < 140 && r.left > 300 && /^\/[A-Za-z0-9_.]+\/?$/.test(h) && !/^\/(direct|explore|reels|accounts)\b/.test(h);
  });
  return links.length ? links[0].getAttribute('href').replace(/\//g, '') : '';
}
"""

# Messages in the open chat, oldest first: {text, mine, y, ref}. "mine" is decided by several signals (their avatar
# beside the bubble, bubble colour, which side of the chat it sits on, row alignment), because no single one is
# reliable across themes and layouts.
MESSAGES = r"""
() => {
  const textbox = document.querySelector("div[role='textbox'][contenteditable='true']") || document.querySelector("div[aria-label*='Message']");
  if (!textbox) return null;
  const tb = textbox.getBoundingClientRect();
  const chatLeft = tb.left, chatRight = tb.right, chatCenter = tb.left + tb.width / 2;
  const top = Math.max(35, tb.top - 700), bottom = tb.top + 5, left = chatLeft - 65, right = chatRight + 65;
  const status = /^(seen|delivered|sent|active|today|yesterday|\d{1,2}:\d{2})/i;
  const label = /^(you replied to .{1,60}|.{1,60} replied to (you|yourself|themselves|themself|himself|herself|your story|their story)|.{1,60} reacted .{1,12} to your message)$/i;
  const out = [];
  const score = (el, rect) => {
    let s = 0;
    const row = el.closest("div[role='row'], div[role='listitem']") || (el.parentElement && el.parentElement.parentElement);
    if (row) {
      for (const av of row.querySelectorAll("img")) {
        const a = av.getBoundingClientRect();
        if (a.width >= 14 && a.width <= 48 && Math.abs(a.width - a.height) <= 4 && a.right <= rect.left + 4) { s -= 10; break; }
      }
    }
    let p = el;
    for (let d = 0; d < 4 && p && p !== document.body; d++) {
      const bg = getComputedStyle(p).backgroundColor;
      const m = bg && bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (m && bg !== 'rgba(0, 0, 0, 0)') {
        const [r, g, b] = [+m[1], +m[2], +m[3]];
        if ((b > r + 25 && b > g + 15) || (b > 160 && g < 140)) { s += 8; break; }
        if (Math.abs(r - g) <= 15 && Math.abs(g - b) <= 15 && ((r > 190) || (r < 80))) { s -= 6; break; }
      }
      p = p.parentElement;
    }
    const c = rect.left + rect.width / 2;
    if (c > chatCenter + 15) s += 6; else if (c < chatCenter - 15) s -= 6;
    const dl = rect.left - chatLeft, dr = chatRight - rect.right;
    if (dr < 40 && dr < dl) s += 4; else if (dl < 80 && dl < dr) s -= 4;
    return s;
  };
  for (const el of document.querySelectorAll("div[dir='auto'], span[dir='auto']")) {
    if (textbox.contains(el) || el.closest("button, header, div[role='banner'], a[role='link']")) continue;
    const text = (el.innerText || "").trim();
    if (!text || status.test(text) || label.test(text)) continue;
    const low = text.toLowerCase();
    if (low.includes("followed by") || low.startsWith("suggested") || low === "message...") continue;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height || r.bottom > bottom || r.top < top) continue;
    if (r.left < left || r.left > right) continue;  // must START inside the chat column (the Notes tray sits left)
    if (out.some(m => (m.text === text || m.text.includes(text)) && Math.abs(m.y - r.top) < 12)) continue;
    out.push({text, mine: score(el, r) > 0, y: r.top, ref: null});
  }
  // shared reels / posts / media: one "[Shared a Reel]" per card, tagged so it can be quote-replied later
  for (const mEl of document.querySelectorAll("a[href*='/reel/'], a[href*='/p/'], video, svg[aria-label*='Clip' i], svg[aria-label*='Reels' i]")) {
    if (textbox.contains(mEl) || mEl.closest("header")) continue;
    const card = mEl.closest("[role='button']") || mEl.closest("a[href]") || mEl;
    if (card.getAttribute("data-rm-seen")) continue;
    card.setAttribute("data-rm-seen", "1");
    const r = card.getBoundingClientRect();
    if (!r.width || !r.height || r.bottom > bottom || r.top < top || r.left < left || r.left > right) continue;
    window.__rmSeq = (window.__rmSeq || 0) + 1;
    const ref = "m" + window.__rmSeq;
    card.setAttribute("data-rm-ref", ref);
    const mine = score(card, r) > 0;
    const near = out.find(m => Math.abs(m.y - r.top) < 30 && m.mine === mine);
    if (near) { near.text = "[Shared a Reel] " + near.text; near.ref = ref; continue; }
    out.push({text: "[Shared a Reel]", mine, y: r.top, ref});
  }
  document.querySelectorAll("[data-rm-seen]").forEach(e => e.removeAttribute("data-rm-seen"));
  out.sort((a, b) => a.y - b.y);
  return out;
}
"""

# where a bubble (by its text) or a reel card (by its ref) is, scrolled into view
FIND_BUBBLE = r"""
([text, ref]) => {
  let el = null;
  if (ref) el = document.querySelector('[data-rm-ref="' + ref + '"]');
  if (!el && text) {
    const norm = s => (s || '').replace(/\s+/g, ' ').trim();
    const tb = document.querySelector("div[role='textbox'][contenteditable='true']");
    const els = [...document.querySelectorAll("div[dir='auto'], span[dir='auto']")]
      .filter(e => !(tb && tb.contains(e)) && norm(e.innerText) === norm(text) && e.getBoundingClientRect().height > 0);
    el = els[els.length - 1];
  }
  if (!el) return null;
  el.scrollIntoView({block: 'center'});
  const r = el.getBoundingClientRect();
  return {x: r.left + r.width / 2, y: r.top + r.height / 2, h: r.height};
}
"""

# the hover toolbar button (Reply / React) nearest to a bubble's row
HOVER_BUTTON = r"""
([pattern, y, maxd]) => {
  const re = new RegExp(pattern, 'i');
  const lim = maxd || 70;
  const btns = [...document.querySelectorAll('[aria-label]')].map(b => {
    const r = b.getBoundingClientRect();
    return {b, r, d: Math.abs((r.top + r.height / 2) - y)};
  }).filter(o => o.r.width > 0 && o.r.height > 0 && o.d < lim && re.test(o.b.getAttribute('aria-label') || ''));
  btns.sort((a, b) => a.d - b.d);
  if (!btns.length) return null;
  const {b, r} = btns[0];
  return {x: r.left + r.width / 2, y: r.top + r.height / 2, label: b.getAttribute('aria-label')};
}
"""

CHECKPOINTS = [
    "form#checkpointSubmitForm", "iframe[src*='recaptcha']", "iframe[src*='hcaptcha']",
    "div:has-text('Suspicious Activity')", "div:has-text('Action Blocked')", "div:has-text('Help us confirm it')",
]
MESSAGE_BOX = ["div[role='textbox'][contenteditable='true']", "div[aria-label='Message']", "div[aria-label='Message...']"]
LOGIN_PAGE = r"/accounts/login|/challenge/|/accounts/suspended|/two_factor"
