"""python -m replace_me <command>

    init        create me/profile.yaml, me/now.md from the templates (won't overwrite)
    ingest      read me/export/ -> your real replies (.data/corpus.*) + every chat (.data/chats.*)
    research    people | me : notes on each person / your master profile (a strong model reads your chats once)
    chat        talk to your copy in the terminal (nothing is sent to Instagram)
    login       open a browser and log in to Instagram yourself (saved in browser_profile/)
    run         reply to your Instagram DMs
    mute        mute|unmute <username>: replies off/on for someone
    eval        blind test: real you vs your copy (see docs/EVALUATION.md)
    finetune    dataset : build the training set for your own model (docs/FINE_TUNING.md)
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys

from replace_me.config import get_settings


def cmd_init() -> None:
    s = get_settings()
    for src, dst in (("profile.example.yaml", "profile.yaml"), ("now.example.md", "now.md"),
                     ("mask_extra.example.txt", "mask_extra.txt"), ("mask_keep.example.txt", "mask_keep.txt")):
        if not (s.ME_DIR / dst).exists() and (s.ME_DIR / src).exists():
            shutil.copy(s.ME_DIR / src, s.ME_DIR / dst)
            print(f"created me/{dst}")
    (s.ME_DIR / "export").mkdir(parents=True, exist_ok=True)
    (s.ME_DIR / "people").mkdir(parents=True, exist_ok=True)
    print("Next: put your Instagram export in me/export/, fill in me/profile.yaml, then `python -m replace_me ingest`.")


def cmd_ingest() -> None:
    from replace_me.memory import chats as chat_mem
    from replace_me.memory import corpus
    from replace_me.memory.export import guess_my_name, read_chats
    from replace_me.profile import load_profile

    s = get_settings()
    profile = load_profile()
    names = set(profile.get("export_names") or ([profile["name"]] if profile.get("name") else []))
    if not names:
        guess = guess_my_name(s.export_dir)
        if not guess:
            sys.exit("No export found in me/export/ (see me/export/README.md).")
        print(f"Your name in the export looks like '{guess}'. Add it to export_names in me/profile.yaml to be sure.")
        names = {guess}
    all_chats = list(read_chats(s.export_dir, names))
    if not all_chats:
        sys.exit("No chats read from me/export/. Is it the JSON format? (me/export/README.md)")
    swears = str(profile.get("swear_words") or corpus.DEFAULT_SWEARS)
    print("your replies:", corpus.build(all_chats, s.DATA_DIR, swears))
    print("chat memory:", chat_mem.build(all_chats, s.DATA_DIR))
    print("Next: `python -m replace_me research people`, then `research me`.")


async def cmd_chat(handle: str) -> None:
    from replace_me.replier import Replier
    from replace_me.store import Store

    s = get_settings()
    r = Replier(Store(s.DATA_DIR / "terminal_chat.db"))
    print(f"Talking to your copy as @{handle} (Ctrl+C to stop). Nothing is sent anywhere but the model.")
    while True:
        try:
            msg = input("them> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not msg:
            continue
        reply = await r.answer(handle, handle, [m.strip() for m in msg.split(" / ") if m.strip()])
        for b in reply.bubbles:
            print("you >", b)
        print(f"      ({round(reply.latency_ms)} ms, {reply.model}, lookups: {', '.join(reply.searches) or 'none'})")


async def cmd_run() -> None:
    from replace_me.instagram.chat import SecurityCheck
    from replace_me.instagram.inbox import Inbox
    from replace_me.instagram.session import launch

    pw, ctx = await launch(headless=False)
    try:
        await Inbox(ctx).run()
    except SecurityCheck as e:
        print(f"\nSTOPPED: {e}\nLook at the browser window and deal with it yourself; the bot never gets around these.")
    finally:
        await ctx.close()
        await pw.stop()


def main() -> None:
    ap = argparse.ArgumentParser(prog="replace_me", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("ingest")
    rp = sub.add_parser("research")
    rp.add_argument("what", choices=["people", "me"])
    rp.add_argument("--only", nargs="*", default=[])
    rp.add_argument("--min", type=int, default=40, help="skip chats where you sent fewer messages than this")
    rp.add_argument("--force", action="store_true")
    cp = sub.add_parser("chat")
    cp.add_argument("--as", dest="handle", default="test_friend", help="the Instagram username to pretend to be")
    sub.add_parser("login")
    sub.add_parser("run")
    mp = sub.add_parser("mute")
    mp.add_argument("action", choices=["mute", "unmute"])
    mp.add_argument("username")
    ep = sub.add_parser("eval")
    ep.add_argument("--n", type=int, default=30)
    ep.add_argument("--seed", type=int, default=7)
    ep.add_argument("--life", action="store_true", help="give the copy your life around each moment (fair: nothing after it)")
    fp = sub.add_parser("finetune")
    fp.add_argument("what", choices=["dataset"])
    a = ap.parse_args()

    if a.cmd == "init":
        cmd_init()
    elif a.cmd == "ingest":
        cmd_ingest()
    elif a.cmd == "research":
        from replace_me import research

        asyncio.run(research.run_people(a.only, a.min, a.force) if a.what == "people" else research.run_me())
    elif a.cmd == "chat":
        asyncio.run(cmd_chat(a.handle))
    elif a.cmd == "login":
        from replace_me.instagram.session import login

        asyncio.run(login())
    elif a.cmd == "run":
        asyncio.run(cmd_run())
    elif a.cmd == "mute":
        from replace_me.store import Store

        Store().mute(a.username, on=a.action == "mute")
        print(f"replies {'off' if a.action == 'mute' else 'on'} for @{a.username.lstrip('@')}")
    elif a.cmd == "eval":
        from replace_me import evaluate

        asyncio.run(evaluate.run(a.n, a.seed, a.life))
    elif a.cmd == "finetune":
        from replace_me import finetune

        finetune.build_dataset()


if __name__ == "__main__":
    main()
