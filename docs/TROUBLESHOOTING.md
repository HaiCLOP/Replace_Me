# Troubleshooting

Each of these happened for real.

## Setup

| Symptom | Fix |
|---|---|
| `ingest`: "No chats read" | The export must be **JSON**, not HTML. Files are `message_1.json`, at any depth under `me/export/`. |
| `ingest` finds 0 replies | `export_names` in `me/profile.yaml` must match your sender name in the export exactly. `ingest` prints its guess when it's empty. |
| Names look like `Ã°Å¸Â˜Â‚` | Instagram's export double-encodes UTF-8. The reader fixes it, so if you see it, you're reading the files with something else. |
| `UnicodeEncodeError: 'charmap'` on Windows | The console can't print an emoji. Set `PYTHONUTF8=1` (or `PYTHONIOENCODING=utf-8`). |

## Instagram

| Symptom | Fix |
|---|---|
| Logging in keeps failing, the page reloads | Run `login` and don't run `run` at the same time. The bot leaves login and challenge pages alone. |
| "STOPPED: security check" | Instagram wants you to confirm something. Do it yourself in the window, then start `run` again. |
| It replies to its own message, or reads the wrong side | A DOM change. The geometry checks in `instagram/dom.py` decide whose bubble is whose. Fix the selector there. |
| Replies feel slow | Most of the time is the model (thinking + 4 candidates). `CANDIDATES=2`, or a lower effort (`@low`) in `LLM_CHAIN`. Several chats are answered in parallel anyway. |
| It answered someone you muted | Mutes are checked before reading and right before sending. If it still happened, check the username in `mute` (no `@`, as in their profile URL). |

## Replies

| Symptom | Fix |
|---|---|
| Too polite, too complete | Fix the person's note, add real lines to `me/now.md`. Don't add style rules (docs/EVALUATION.md). |
| Mostly one-word replies | `PICKER=judge` limits replies to your real length mix. Check `.data/stats.json` → `length_mix`. |
| It never mentions anything about your life | `me/now.md` is empty or vague. Write specific things ("stuck on the contact form"), not "busy". |
| It says "from our earlier chat..." | That narration is filtered. If a new phrasing gets through, add it to `_NARRATION` in `brain/filters.py`. |
| Tool calls end with no reply | One nudge is built in. If it keeps happening, the model chain's first link may not support tools: put an Anthropic model first. |

## Fine-tuning

| Symptom | Fix |
|---|---|
| Kaggle: `No module named 'unsloth'` | The notebook had no internet. Verify your phone on Kaggle and set Internet **On**. |
| Kaggle finished, no `.gguf` in Output | It was written to `/tmp`. Use `my_lora/` and `finetune lora-to-gguf` (docs/FINE_TUNING.md). |
| `ollama create`: "no Modelfile or safetensors files found" | Ollama can't load a Qwen3 safetensors adapter. Convert it with `finetune lora-to-gguf` first. |
| The voice writes `[sends a reel]` | It learned media placeholders from your export. They're dropped before sending. |
| First reply takes ~20 s | Ollama loads the model on first use. Later replies take 2-5 s. |
