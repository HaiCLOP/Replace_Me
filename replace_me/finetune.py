"""Your own model: a small open model (Qwen3-8B) fine-tuned on your real replies, run on your PC with Ollama.

    python -m replace_me finetune dataset      # -> finetune/out/{train,val,heldout}.jsonl + kaggle_train.ipynb + Modelfile
    python -m replace_me finetune lora-to-gguf # adapter folder from Kaggle -> finetune/out/my-lora.gguf (fallback)

The dataset is your chats, so finetune/out/ is git-ignored. Upload it only to a PRIVATE Kaggle dataset.
Full walkthrough (free Kaggle GPU, ~5 hours for 16k replies): docs/FINE_TUNING.md.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from replace_me.config import ROOT, get_settings
from replace_me.memory.export import Chat, read_chats
from replace_me.profile import load_profile

OUT = ROOT / "finetune" / "out"
CONTEXT = 24                # messages before your reply the model sees
HOLDOUT_DAYS = 14           # the last two weeks of every chat are never trained on (fair tests stay fair)
TURN_GAP_MS = 10 * 60 * 1000
SYSTEM = ("You are {name}, texting {who} on Instagram from your phone. Reply exactly the way you really do: your "
          "words, spelling, slang, length and emoji. One message per line.")
# lines that would teach the model to talk about being an AI (you were probably testing a bot in some chats)
_ABOUT_AI = re.compile(r"\b(bot|chatgpt|claude|gpt|openai|llm|ai\s+(?:hai|h|is|reply)|auto.?reply|prompt)\b", re.I)
_PRIVATE = [re.compile(r"\b\d{10}\b"), re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)]


def gap_label(prev_ms: int | None, ms: int) -> str:
    if prev_ms is None:
        return ""
    h = (ms - prev_ms) / 3_600_000
    if h >= 24:
        return f"({int(h // 24)} day{'s' if h >= 48 else ''} later)"
    if h >= 1:
        return f"({int(h)} hour{'s' if h >= 2 else ''} later)"
    return ""


def scrub(t: str) -> str:
    for p in _PRIVATE:
        t = p.sub("[number]", t)
    return t


def render(chat: Chat, start: int, end: int) -> str:
    lines, prev = [], None
    for m in chat.messages[start:end]:
        g = gap_label(prev, m.ts)
        if g:
            lines.append(g)
        prev = m.ts
        lines += [f"{'You' if m.mine else 'Them'}: {scrub(x.strip())}" for x in m.text.splitlines() if x.strip()]
    return "\n".join(lines)


def reply_turns(chat: Chat) -> list[tuple[int, int]]:
    """(first, last+1) of every run of your messages that answers theirs."""
    ms, out, i = chat.messages, [], 1
    while i < len(ms):
        if ms[i].mine and not ms[i - 1].mine:
            j = i
            while j + 1 < len(ms) and ms[j + 1].mine and ms[j + 1].ts - ms[j].ts < TURN_GAP_MS:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def examples(chats: list[Chat], name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    train, held, stats = [], [], Counter()
    for chat in chats:
        if chat.group or not chat.messages:
            continue
        cutoff = chat.messages[-1].ts - HOLDOUT_DAYS * 86_400_000
        who = chat.names[0] if chat.names else "a friend"
        for i, j in reply_turns(chat):
            reply = [scrub(x.strip()) for m in chat.messages[i:j] for x in m.text.splitlines() if x.strip()]
            context = render(chat, max(0, i - CONTEXT), i)
            if not reply or not context:
                continue
            if _ABOUT_AI.search(" ".join(reply)):
                stats["dropped_ai_talk"] += 1
                continue
            row = {"messages": [{"role": "system", "content": SYSTEM.format(name=name, who=who)},
                                {"role": "user", "content": context + "\n\nReply as you:"},
                                {"role": "assistant", "content": "\n".join(reply[:5])}]}
            (held if chat.messages[i].ts >= cutoff else train).append(row)
            stats["held_out" if chat.messages[i].ts >= cutoff else "train"] += 1
    return train, held, stats


def build_dataset(seed: int = 3407) -> None:
    s = get_settings()
    profile = load_profile()
    names = set(profile.get("export_names") or [profile.get("name", "")])
    chats = list(read_chats(s.export_dir, names))
    if not chats:
        sys.exit("No chats in me/export/ (see me/export/README.md).")
    train, held, stats = examples(chats, str(profile.get("name") or "me"))
    random.Random(seed).shuffle(train)
    n_val = max(50, len(train) // 50)
    val, train = train[:n_val], train[n_val:]
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("train", train), ("val", val), ("heldout", held)):
        with (OUT / f"{fname}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_notebook(str(profile.get("name") or "me"))
    write_modelfile()
    print(f"train {len(train)}, val {len(val)}, held out {len(held)}, dropped (talk about AI) {stats['dropped_ai_talk']}")
    print(f"-> {OUT}. Next: docs/FINE_TUNING.md (upload train.jsonl + val.jsonl as a PRIVATE Kaggle dataset).")


NOTEBOOK_CELLS = [
    ("markdown", """# Fine-tune Qwen3-8B on your replies (free Kaggle GPU)

Settings (right panel): Accelerator **GPU T4 x2**, Internet **On** (needs a phone-verified Kaggle account).
**+ Add Input** your private dataset with `train.jsonl` and `val.jsonl`. Then **Run All** / **Save Version**.
At the end download `my-voice-q4.gguf` (~5 GB) from Output. `my_lora/` is the small trained part, kept as a backup."""),
    ("code", """# 1. install (Kaggle has torch/CUDA); one GPU: Unsloth can trip over the second T4
import os, glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
!pip install -q --upgrade unsloth unsloth_zoo
TRAIN = glob.glob('/kaggle/input/**/train.jsonl', recursive=True)
VAL = glob.glob('/kaggle/input/**/val.jsonl', recursive=True)
assert TRAIN, 'Add your dataset with train.jsonl (+ Add Input on the right)'"""),
    ("code", """# 2. settings
MODEL = 'unsloth/Qwen3-8B'   # any ~7-8B Qwen from unsloth works
MAX_LEN = 1536                # fits a 16 GB T4
EPOCHS = 1
MAX_STEPS = -1                # e.g. 600 to cap the time"""),
    ("code", """# 3. 4-bit model + LoRA
from unsloth import FastLanguageModel, is_bfloat16_supported
model, tokenizer = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAX_LEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias='none',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    use_gradient_checkpointing='unsloth', random_state=3407)"""),
    ("code", """# 4. data, in the model's own chat template
from datasets import load_dataset
ds = load_dataset('json', data_files={'train': TRAIN[0], 'val': VAL[0]})
ds = ds.map(lambda ex: {'text': tokenizer.apply_chat_template(ex['messages'], tokenize=False)}, remove_columns=['messages'])"""),
    ("code", """# 5. train only on YOUR replies (not on the chat before them)
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds['train'], eval_dataset=ds['val'],
    args=SFTConfig(dataset_text_field='text', max_seq_length=MAX_LEN, packing=True,
        per_device_train_batch_size=2, gradient_accumulation_steps=8, num_train_epochs=EPOCHS, max_steps=MAX_STEPS,
        warmup_steps=20, learning_rate=2e-4, lr_scheduler_type='linear', optim='adamw_8bit', weight_decay=0.01,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=25,
        eval_strategy='steps', eval_steps=200, save_strategy='no', output_dir='outputs', report_to='none', seed=3407))
trainer = train_on_responses_only(trainer, instruction_part='<|im_start|>user\\n', response_part='<|im_start|>assistant\\n')
print(trainer.train())"""),
    ("code", """# 6. does it sound like you?
FastLanguageModel.for_inference(model)
msgs = [{'role': 'system', 'content': '__SYSTEM__'}, {'role': 'user', 'content': 'Them: hey\\nThem: what are you up to\\n\\nReply as you:'}]
ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to('cuda')
print(tokenizer.decode(model.generate(input_ids=ids, max_new_tokens=40, temperature=0.9, do_sample=True)[0][ids.shape[1]:], skip_special_tokens=True))"""),
    ("code", """# 7. save: the small LoRA first, then the GGUF for Ollama.
# Newer Unsloth writes the GGUF under /tmp/unsloth_saves, which Kaggle throws away: copy it into /kaggle/working.
import shutil
model.save_pretrained('my_lora'); tokenizer.save_pretrained('my_lora')
model.save_pretrained_gguf('my-voice', tokenizer, quantization_method='q4_k_m')
found = [g for g in glob.glob('my-voice*/*.gguf') + glob.glob('/tmp/unsloth_saves/**/*.gguf', recursive=True) if 'Q4' in g.upper()]
for f in found[:1]:
    shutil.copy(f, '/kaggle/working/my-voice-q4.gguf'); print('saved', f)
assert found, 'no GGUF written: download my_lora/ and run `python -m replace_me finetune lora-to-gguf`'
!ls -la /kaggle/working"""),
]

MODELFILE = '''# ollama create my-voice -f Modelfile   (run in the folder with the .gguf)
FROM ./my-voice-q4.gguf
# Fallback when only my_lora/ came back from Kaggle (see docs/FINE_TUNING.md):
# FROM qwen3:8b
# ADAPTER ./my-lora.gguf
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.8
PARAMETER num_ctx 2048
'''


def write_notebook(name: str) -> Path:
    system = SYSTEM.format(name=name, who="a friend").replace("'", "\\'")
    nb = {"cells": [{"cell_type": t, "metadata": {}, "source": src.replace("__SYSTEM__", system).splitlines(keepends=True),
                     **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, src in NOTEBOOK_CELLS],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = OUT / "kaggle_train.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return path


def write_modelfile() -> Path:
    path = OUT / "Modelfile"
    path.write_text(MODELFILE, encoding="utf-8")
    return path


# PEFT target module -> llama.cpp tensor name (Qwen/Llama layout)
_GGUF_NAMES = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output",
               "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}
_LORA_KEY = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])\.weight$")


def lora_to_gguf(lora_dir: Path, out: Path, arch: str = "qwen3") -> int:
    """A PEFT LoRA adapter -> a llama.cpp LoRA GGUF, which Ollama loads with `ADAPTER` on top of the base model.
    Ollama can't read a Qwen3 safetensors adapter itself. Needs `pip install gguf safetensors`."""
    import numpy as np
    try:
        import gguf
        from safetensors import safe_open
    except ImportError:
        sys.exit("pip install gguf safetensors   (only needed for this step)")
    cfg = json.loads((lora_dir / "adapter_config.json").read_text(encoding="utf-8"))
    w = gguf.GGUFWriter(str(out), arch)
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(cfg["lora_alpha"]))
    f = safe_open(str(lora_dir / "adapter_model.safetensors"), "np")
    n = 0
    for key in sorted(f.keys()):
        m = _LORA_KEY.search(key)
        if not m or m.group(2) not in _GGUF_NAMES:
            sys.exit(f"unexpected tensor {key}: is this a LoRA on a Qwen/Llama-style model?")
        layer, mod, ab = m.groups()
        w.add_tensor(f"blk.{layer}.{_GGUF_NAMES[mod]}.weight.lora_{ab.lower()}", f.get_tensor(key).astype(np.float32))
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n
