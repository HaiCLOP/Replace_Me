# CLAUDE.md

Read [AGENTS.md](AGENTS.md) first: it's the guide for any AI agent working on this repo. In short:

- Never stage anything from `me/`, `.data/`, `reports/`, `browser_profile/`, `finetune/out/`, `.env` or `*.gguf`,
  and never write a real person's name, username or chat line anywhere in the repo.
- Changes to how replies are written need a blind test with the same seed before and after
  (`python -m replace_me eval --n 30 --seed 7`). +-2-3 out of 30 is noise.
- `pytest -q` is offline: no model calls, no network, no browser.
- Keep the safety behaviour: name masking for cloud models, honest bot answer, no money/OTP asks, gossip only
  for 90+, stop on security checks, mute re-checks, no message requests.
