"""Offline tests on a made-up export (tests/fixtures/export): no model calls, no network, no browser."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from replace_me import evaluate, finetune
from replace_me.brain import filters
from replace_me.brain.lookups import dates_in
from replace_me.llm.masking import NameMasker
from replace_me.memory import corpus as corpus_mod
from replace_me.memory.export import read_chats

FIXTURE = Path(__file__).parent / "fixtures" / "export"
ME = {"Alex Example"}


@pytest.fixture
def chats():
    return list(read_chats(FIXTURE, ME))


@pytest.fixture
def fake_embed(monkeypatch):
    """Deterministic stand-in for the local embedding model (so tests don't download it)."""
    def embed(texts):
        out = np.zeros((len(texts), 16), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w) % 16] += 1
            out[i] /= np.linalg.norm(out[i]) or 1
        return out
    monkeypatch.setattr(corpus_mod, "embed", embed)
    return embed


def test_export_is_read_oldest_first_and_mine_is_marked(chats):
    by = {c.folder: c for c in chats}
    assert set(by) == {"sam_10001", "jordan_10002"}
    sam = by["sam_10001"]
    assert sam.names == ["Sam Rivers"] and not sam.group
    assert [m.ts for m in sam.messages] == sorted(m.ts for m in sam.messages)
    assert sam.messages[0].text == "hey" and not sam.messages[0].mine
    assert any(m.mine and m.text == "yesss" for m in sam.messages)


def test_corpus_pairs_their_message_with_your_reply(chats, fake_embed, tmp_path):
    info = corpus_mod.build(chats, tmp_path)
    assert info["examples"] >= 5
    c = corpus_mod.Corpus(tmp_path).load()
    first = next(e for e in c.examples if e.reply[0] == "yesss")
    assert first.reply == ["yesss", "football at 5 right"]      # one turn = your messages close together
    assert first.context[-1]["who"] == "them"
    assert not any("9876543210" in " ".join(e.reply) or "9876543210" in e.key for e in c.examples if e.reply == ["ok calling"])
    held = first.key
    assert all(e.key != held for e in c.find("you coming sunday", k=5, n_random=0, exclude_keys=[held]))
    assert c.stats["messages"] > 0 and len(c.stats["length_mix"]) == 4


def test_masking_round_trip_and_new_people():
    m = NameMasker(names=["Sam Rivers", "Jordan Lee"], handles=["sam.k"])
    masked = m.mask("Sam said @sam.k and jordan are coming")
    assert "Sam" not in masked and "sam.k" not in masked and "jordan" not in masked.lower()
    assert m.unmask(masked).lower() == "sam said @sam.k and jordan are coming"
    assert m.names_new_people("ask jordan", allowed="sam")
    assert not m.names_new_people("ask sam", allowed="sam")


def test_filters_keep_replies_human():
    assert filters.clean_raw("You: hey\nThem: what\nmore") == ["hey"]
    assert filters.fit(["a?", "b?", "c", "d"], 3) == ["a?", "c", "d"]           # one question at most
    assert filters.drop_money_asks(["send me 500 rupees?", "lol"]) == ["lol"]
    assert filters.drop_echoes(["good night sweet dreams", "ok"], ["good night sweet dreams"]) == ["ok"]
    opts = [["k"], ["ok see you there"], ["bro that is honestly so funny man i cant"]]
    assert filters.by_your_length(opts, [0, 1, 0, 0], random.Random(1)) == [["ok see you there"]]


def test_dates_in_messages():
    assert dates_in("what happened on 17th april") == [(4, 17, None)]
    assert dates_in("april 17, 2026") == [(4, 17, 2026)]
    assert dates_in("on 2026-04-17") == [(4, 17, 2026)]


def test_finetune_examples_drop_ai_talk_and_numbers(chats):
    train, held, stats = finetune.examples(chats, "Alex")
    rows = train + held
    assert rows and all(r["messages"][0]["role"] == "system" and r["messages"][2]["role"] == "assistant" for r in rows)
    replies = [r["messages"][2]["content"] for r in rows]
    assert "yesss\nfootball at 5 right" in replies
    assert all("9876543210" not in r["messages"][1]["content"] for r in rows)
    assert all(r["messages"][1]["content"].endswith("Reply as you:") for r in rows)


def test_notebook_saves_gguf_out_of_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(finetune, "OUT", tmp_path)
    nb = json.loads(finetune.write_notebook("Alex").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "/tmp/unsloth_saves" in code and "/kaggle/working/my-voice-q4.gguf" in code
    assert "Alex" in code
    assert "ADAPTER" in finetune.write_modelfile().read_text(encoding="utf-8")


def test_eval_helpers():
    assert evaluate.squash("Heyyy!!") == evaluate.squash("hey")
    st = evaluate.stats_of([["hi"], ["what?", "ok 😂"]])
    assert st["messages per reply"] == 1.5 and st["replies with a question %"] == 50
    assert evaluate.leaks("he said BRO NO lol", ["bro no"])


async def test_judge_is_scored_against_the_hidden_side():
    class Fake:
        def __init__(self, answer):
            self.answer = answer

        async def generate(self, *a, **k):
            return type("R", (), {"content": f"REAL: {self.answer}\nTELL: too neat"})()

    for answer in "AB":
        rng = random.Random(0)
        copy_first = random.Random(0).random() < 0.5
        fooled, tell = await evaluate.judge(Fake(answer), [], [], ["real"], ["copy"], rng)
        assert fooled == ((answer == "A") == copy_first) and tell == "too neat"
