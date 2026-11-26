# Architecture

## Data flow

```
me/export/ ──ingest──► .data/corpus.jsonl + .npy   your replies, each with what it answered (+ embedding of their message)
            │          .data/stats.json            your numbers: burst sizes, length mix, emoji/question/swear rates
            └────────► .data/chats.jsonl + .npy    every chat, 10-message chunks every 6 messages, embedded
me/export/ ──research─► me/people/<chat>.md         how you are with each person
                        me/people/_me.md            how you are with everyone
live DMs ──────────────► .data/live.db              what the bot saw and sent since the export (searchable too)
```

## One reply

`instagram/inbox.py` sees a chat with something new, opens it in its own tab (up to `INBOX_MAX_PARALLEL`), waits
`BURST_SETTLE_SECONDS` for them to finish typing, and reads the new bubbles. Then `Replier.answer(handle, name,
messages)`:

1. **Situation** (`replier.py`). Who they are (`me/profile.yaml`), whether they're close enough for gossip,
   their note, your latest real lines with them, what you've typed in other chats these days, and the
   **lookups** this person may use:
   - `them`: search your chat with them
   - `everyone`: search every chat (only for scores ≥ `GOSSIP_MIN_SCORE`)
   - `mine`: what you said anywhere
   - `day`: everything said on a date
   - `examples`: more of your real replies
2. **Always looked up** (`brain/lookups.py`). Dates in their message ("17th april") and story asks ("what
   happened with...") are searched before the model is called, so the answer doesn't depend on it choosing to.
3. **Prompt** (`brain/prompt.py`, `brain/request.py`).
   - The cached part: your profile, master profile, life now, typing facts and the rules, then the person block,
     then the chat as real user/assistant turns. It uses Anthropic prompt caching, so a hot chat costs a fraction
     of the first message.
   - The fresh part: 16 of your real replies to the most similar messages (minus ones you sent lately), what's
     found, and how many messages to send (drawn from your real burst sizes).
4. **Write** (`brain/reply.py`). The model (`LLM_CHAIN`, Opus with thinking by default) may call the lookup tools
   twice, then writes `CANDIDATES` replies. With no tool-capable model it falls back to one plain prompt.
   - Everything sent to a cloud model is name-masked (`llm/masking.py`), and the answer is unmasked.
5. **Clean** (`brain/filters.py`). Drop labels and narration ("from our earlier chat..."), thin the swearing to
   your real rate (none with `clean` people), drop lines that just echo their words, never ask for money, OTPs
   or passwords, and keep at most one question.
6. **Option C** (`brain/select.py: in_your_voice`, `VOICE=meaning_all`). Your fine-tuned model on your PC
   rewrites each candidate in your words, given the chat and the meaning. Lines that name someone new are
   dropped, and so are `[photo]`-style placeholders.
7. **Pick** (`brain/select.py: pick_like_a_judge`). Keep only candidates near a length drawn from your real
   length mix, then a model that thinks like the blind judge picks the one most like you.
8. **Send** (`instagram/chat.py`). A human reading pause and typing speed. A reply to a reel quotes that reel.
   Only one tab types at a time. Mutes are re-checked right before sending.
9. **Hot chat.** The tab stays open for `INBOX_HOT_SECONDS`, re-read every `INBOX_HOT_POLL_SECONDS`.

Anything you typed yourself in a chat (bubbles the bot never sent) is saved as yours, so the copy keeps up.

## Modules

| Module | What it does |
|---|---|
| `llm/base.py` | `LLMResponse`, errors, the tool-call callback type |
| `llm/anthropic.py` | Messages API: thinking (`@low/@medium/@high`), prompt caching (1 h), tool loop with one nudge if it ends without text |
| `llm/ollama.py` | Local models (`/api/chat`), `local = True` so nothing is masked |
| `llm/chain.py` | `"a,b,c"` failover chain; masks for cloud links, unmasks answers and tool arguments |
| `llm/masking.py` | Stable codes for every name/username in your export (+ `mask_extra.txt`, minus `mask_keep.txt`) |
| `memory/export.py` | Reads Instagram JSON (any depth), fixes its broken UTF-8, drops system lines |
| `memory/corpus.py` | Your replies: turns, pairs, private-info scrub, stats, similarity search |
| `memory/chats.py` | Hybrid search (embeddings + literal words + recency), by date, "your life around a moment" |
| `memory/embeddings.py` | Local `fastembed` model (BAAI/bge-small-en-v1.5), no API |
| `brain/*` | See "One reply" above |
| `store.py` | SQLite: messages since the export, mutes, which gossip was already told to whom |
| `research.py` | Person notes and master profile, with per-chat numbers so the model can't flatter you |
| `evaluate.py` | Blind test |
| `finetune.py` | Training set, Kaggle notebook, Modelfile, LoRA → GGUF |
| `instagram/session.py` | Persistent Chromium profile; `login` waits for you |
| `instagram/dom.py` | JS that reads bubbles, finds the reply button, quotes a reel (all the DOM guesswork in one place) |
| `instagram/chat.py` | One chat tab: read, quote, type, send; raises `SecurityCheck` on login/challenge pages |
| `instagram/inbox.py` | The watcher: scan, parallel tasks, hot follow-ups, learning what you typed yourself |

## Choices worth knowing

- **Tools rather than one huge prompt.** A cached prompt plus lookups beat stuffing every possibly-relevant chunk
  into the prompt. It's cheaper per message, and the model searches when it needs to.
- **Candidates + a judge-style picker** beat writing one reply at a higher temperature. The length limit is what
  keeps the picker honest.
- **Notes are written once and edited by you.** Re-reading whole chats on every reply is slow and expensive, and
  a note you've corrected is better than anything regenerated.
