# Setup

About an hour, most of it waiting for Instagram's export email.

## 1. Install

```bash
git clone https://github.com/HaiCLOP/Replace_Me.git && cd Replace_Me
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
playwright install chromium
pytest -q                         # offline check: should all pass
```

## 2. Keys

```bash
cp .env.example .env
```

Put your own `ANTHROPIC_API_KEY` in `.env`. Every setting is explained in `.env.example`. The defaults are the
best measured setup (docs/EVALUATION.md), so leave them for a first run.

If you use an Anthropic-compatible gateway instead of the official API, set `ANTHROPIC_BASE_URL`. A gateway can
read everything you send it: read docs/PRIVACY.md first.

## 3. Your export

Follow `me/export/README.md`: Instagram → Accounts Center → Download your information → **Messages only, JSON,
all time**. Unzip it into `me/export/`.

## 4. Your profile

```bash
python -m replace_me init        # me/profile.yaml, me/now.md, mask lists
```

Fill in `me/profile.yaml` (the example explains every field). The parts that matter most:

- `export_names`: exactly how **your** name appears as the sender in the export. Wrong = no replies found.
- `people`: the people you text, with a `score` 1-100. 90+ may hear about your other chats. `clean: true` for
  people you never swear with (family). You don't have to list everyone: anyone missing gets the default.
- `family`, `never`: hard rules.
- `bot_aware_handles`: your own test account(s), which skip the "yes I'm a bot" answer.

Then `me/now.md`: five or so short lines about your life right now. Change it whenever things change; no restart.

## 5. Build the memory

```bash
python -m replace_me ingest
```

Reads the export and writes `.data/`: every reply you sent paired with what it answered, your typing numbers
(reply lengths, emoji, swearing rate), and all chats in searchable chunks. The first run downloads a small local
embedding model (~130 MB). Nothing goes to a cloud model at this step.

## 6. Research

```bash
python -m replace_me research people          # every 1:1 chat where you sent 40+ messages
python -m replace_me research people --only sam_12345
python -m replace_me research me              # after people: the master profile
```

A strong model reads each chat once (masked names) and writes `me/people/<chat>.md`. **This is the expensive
step**: a long chat is up to ~120k input tokens. Start with `--only` on your 5-10 most important chats, or raise
`--min`. Read a few notes and correct anything wrong.

## 7. Try it in the terminal

```bash
python -m replace_me chat --as sam_12345      # pretend to be that person; "a / b" sends two messages
```

Each answer shows the time, the model and which lookups it used. If it sounds wrong, fix the person's note or
`me/now.md` first, before you touch code.

## 8. Measure it

```bash
python -m replace_me eval --n 30 --seed 7
```

About 30 × 6 model calls. The report goes to `reports/` with every moment side by side and what gave the copy
away. See docs/EVALUATION.md.

## 9. Go live

```bash
python -m replace_me login     # a browser opens; log in yourself (2FA included), then close it
python -m replace_me run
```

Keep the window open (minimised is fine). It reads the inbox every 5 s, answers up to 3 chats at once, and keeps
an active chat open for 2 minutes. Message requests are never opened. To stop it for someone:
`python -m replace_me mute mute their_username`.

If Instagram shows a login or security page, the bot stops and leaves it to you.
