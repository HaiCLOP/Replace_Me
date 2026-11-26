# Fine-tuning your own model (option C)

A Qwen3-8B fine-tuned on your replies, running on your PC with Ollama. In the setup that scored 15/30, Claude
still decides *what* to say (it reads the notes, searches your chats, writes 4 candidates) and your model words
each one the way you would. Then the picker chooses.

**Cost:** free (Kaggle gives ~30 GPU hours a week). **Time:** ~5 hours on a T4 for ~16k replies, plus ~10 minutes
of your time. **Your PC:** ~6 GB disk, 8 GB+ RAM. A GPU helps, but 2-5 s per reply on a laptop is fine.

## 1. Dataset

```bash
python -m replace_me finetune dataset
```

Writes `finetune/out/` (git-ignored, it's your chats):

- `train.jsonl`, `val.jsonl`: each of your reply turns as a chat. System prompt, the 24 messages before it with
  gaps like "(3 hours later)", then your real reply, one message per line.
- `heldout.jsonl`: the last 14 days of every chat, never trained on, for a fair test.
- `kaggle_train.ipynb`, `Modelfile`.

Phone numbers and emails are blanked. Replies where you talked about AI or bots are dropped (you were probably
testing). Group chats are skipped.

## 2. Train on Kaggle

1. Make a Kaggle account and **verify your phone**. Without it the notebook has no internet and fails at
   `pip install` with `No module named 'unsloth'`.
2. **Datasets → New dataset**. Upload `train.jsonl` and `val.jsonl`, and set it to **Private**.
3. **Code → New notebook → File → Import notebook**: `finetune/out/kaggle_train.ipynb`.
4. Right panel: Accelerator **GPU T4 x2**, Internet **On**, **+ Add Input** → your dataset.
5. **Save Version → Save & Run All (Commit)**. You can close the tab; it runs in the background.

The Kaggle CLI works too (`pip install kaggle`, then `kaggle datasets create` and `kaggle kernels push`), with
`"enable_gpu": true, "enable_internet": true` in `kernel-metadata.json`.

## 3. Get the model

From the notebook's **Output**: `my-voice-q4.gguf` (~5 GB). Then, in that folder, next to `finetune/out/Modelfile`:

```bash
ollama create my-voice -f Modelfile
ollama run my-voice
```

### If there's no .gguf, only `my_lora/`

Some Unsloth versions write the GGUF under `/tmp`, which Kaggle throws away. The notebook copies it out, but if
it's still missing, the small adapter (`my_lora/`, ~175 MB) is enough:

```bash
pip install gguf safetensors
python -m replace_me finetune lora-to-gguf --lora path/to/my_lora --out finetune/out/my-lora.gguf
ollama pull qwen3:8b
```

In the `Modelfile`, swap the `FROM` line for the two commented fallback lines (`FROM qwen3:8b` and
`ADAPTER ./my-lora.gguf`), then run `ollama create my-voice -f Modelfile`. Ollama can't load a Qwen3 adapter
straight from safetensors ("no Modelfile or safetensors files found"). That's why it's converted first.

## 4. Turn it on

In `.env`:

```
VOICE=meaning_all
VOICE_MODEL=ollama:my-voice
```

Then measure it: `python -m replace_me eval --n 30 --seed 7`, against the same seed without `VOICE`.

## Will it leak other chats?

It memorised your chats, so it could bring up someone from another chat. Every line it writes is checked: if it
names a person who isn't in this chat or in the meaning it was given, its wording is thrown away and the cloud
model's is used. It also can't send media, so `[photo]` lines are dropped.

## Tuning

- `MAX_STEPS=600` in the notebook caps the time if your Kaggle session is short.
- One epoch was enough. Two made it repeat favourite lines.
- Smaller base models (Qwen 3B/4B) are faster on a laptop and noticeably worse at following the meaning.
