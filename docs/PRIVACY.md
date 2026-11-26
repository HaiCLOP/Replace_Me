# Privacy

Your chats are mostly other people's messages. Treat them that way.

## What stays on your PC

- `me/` (your export, profile, notes on people), `.data/` (replies, chat memory, live messages), `reports/`,
  `browser_profile/` (your Instagram login), `finetune/out/`. All git-ignored: you can't push them by accident.
- The embedding model for search runs locally.
- Your fine-tuned model (option C) runs locally with Ollama.

## What goes to the cloud model

Each reply sends the model your profile, the person's note, recent messages in that chat, some of your old replies
as examples, and whatever its searches find. Research sends whole chats, once each.

Before anything is sent, **every name and username from your export is swapped for a stable code** (`Pk3f`,
`@Hq9a`), and swapped back in the answer. Add more names to `me/mask_extra.txt`, and words that aren't names to
`me/mask_keep.txt`. What masking doesn't hide: what people said, places, schools, events. A determined reader can
still work out a lot from context.

- **Anthropic's API:** by default API data isn't used for training. Check their current policy.
- **A third-party gateway** (`ANTHROPIC_BASE_URL`) can read and keep everything. Use one only if you trust whoever
  runs it as much as you trust Anthropic.
- **Kaggle:** only if you fine-tune. Upload the dataset as **Private**, and delete it after training if you like.

## What the copy will and won't tell people

- Other chats only for people you scored ≥ `GOSSIP_MIN_SCORE` (90), and each bit of news is told once.
- Never money, OTPs or passwords.
- If someone sincerely asks whether it's a bot, it says yes (except your own test accounts in `bot_aware_handles`).
- `mute` anyone who shouldn't be answered. `IGNORE_HANDLES` for people never answered at all.

## Before you publish a fork

```bash
git status                       # nothing from me/, .data/, reports/, finetune/out/
git grep -n -i "<your name>"     # and your friends' names, school, city
```

Don't put real names or chat lines in issues, test fixtures or commit messages. Tests use made-up people.
