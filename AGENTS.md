# AGENTS.md: notes for AI coding agents

You're working on **Replace Me**, a tool that builds an AI copy of a person from their Instagram chat export and
lets it answer their DMs. Read this before changing anything. Humans: start with the README.

## Ground rules

1. **Never commit personal data.** `me/` (export, profile, notes on people), `.data/`, `reports/`,
   `browser_profile/`, `finetune/out/`, `.env` and `*.gguf` are git-ignored for a reason. Before any commit, run
   `git status` and check that only code, docs, templates and `tests/fixtures/` (made-up data) are staged. Never
   put a real person's name, username, school, city, phone number or chat line in code, docs, tests or commit
   messages. Use made-up examples (Alex, Sam, Jordan).
2. **Measure, don't guess.** Any change to how replies are written (`replace_me/brain/`, prompts, the picker, the
   example selection) needs a before/after blind test with the same seed:
   `python -m replace_me eval --n 30 --seed 7`. With 30 moments the score moves by +-2-3 on its own, so a +1 is
   noise. Write the result in the PR.
3. **Tests run offline.** `pytest` must not call a model, the network or a browser. Use `tests/fixtures/export`
   and monkeypatch `embed` (see `tests/test_pipeline.py`).
4. **Safety behaviour is not optional.** Keep these working, and add a test if you touch them:
   - names and usernames masked for cloud models (`llm/masking.py`, `llm/chain.py`), real text only for local
     (`local = True`) models
   - honest answer to "are you a bot" unless the account is in `bot_aware_handles`
   - never asking for money, OTPs or passwords (`brain/filters.py: drop_money_asks`)
   - other chats only for people scored `>= GOSSIP_MIN_SCORE`
   - stop on Instagram login and security pages, never work around them (`instagram/chat.py: SecurityCheck`)
   - muted people re-checked right before sending
   - message requests never opened
5. **The person's real style wins over "good writing".** Don't add rules that make replies tidier, more polite or
   more complete. The blind judge catches tidy. The copy should be as messy, short and specific as the person.

## Map

```
replace_me/
  cli.py              commands (python -m replace_me ...)
  config.py           settings from .env, data paths
  profile.py          me/profile.yaml, me/now.md, notes in me/people/
  store.py            SQLite: live messages, mutes, what gossip was told to whom
  replier.py          one burst from one person -> Situation -> reply, remembered
  research.py         notes on each person + master profile (one strong-model call per chat)
  evaluate.py         blind test (real vs copy, judge model)
  finetune.py         dataset + Kaggle notebook + Modelfile; LoRA -> GGUF converter
  llm/                Anthropic (caching, tools, thinking) and Ollama clients, failover chain, name masking
  memory/             export reader, your replies (corpus), all chats (search), local embeddings
  brain/              prompt, lookup tools, cached request, filters, candidate picker, voice (option C), write_reply
  instagram/          Playwright session, DOM scripts, one chat tab, the inbox watcher
tests/                offline tests + a made-up export
me/                   the person's data (ignored) + templates
docs/                 SETUP, ARCHITECTURE, EVALUATION, FINE_TUNING, PRIVACY, TROUBLESHOOTING
```

How one reply is made: `docs/ARCHITECTURE.md`. Why each default is what it is: `docs/EVALUATION.md`.

## Commands you'll use

```bash
pip install -e ".[dev]"
pytest -q                                   # offline, a few seconds
python -m replace_me chat --as test_friend  # talk to the copy (needs ingest + an API key)
python -m replace_me eval --n 30 --seed 7   # costs ~30 x (CANDIDATES + 2) model calls
```

## Things that look wrong but aren't

- **Typos, missing punctuation and lowercase in prompts and examples are on purpose.** They're the person's
  real messages.
- **`by_your_length` limits the picker.** Left free, a judge-style picker picks one-word replies 83-90% of the
  time against a real 63%. That fools a judge once and makes the copy dull.
- **Held-out replies are removed from the examples during eval** (`exclude_keys`). Without that, the copy finds
  its own answer among the similar examples and the score means nothing.
- **The "Real exchanges" section of a person's note is never shown to the reply model.** A model pastes those
  lines word for word.
- **The fine-tuned model can't bring in new names** (`masker.names_new_people`). It memorised other chats, and
  it would pull in someone else's story.
- **Instagram DOM selectors live in `instagram/dom.py` as JS strings.** Instagram changes its markup often.
  When something breaks there, fix the selector and keep the geometry checks (e.g. a bubble must start inside
  the chat column).

## Style

Python 3.11+, type hints, small functions, docstrings that say *why*. Log with `replace_me.log.get_logger` as
`area.event key=value`. No new dependencies without a reason. Optional features (fine-tuning) import their
packages inside the function.
