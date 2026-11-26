# Evaluation

## The blind test

```bash
python -m replace_me eval --n 30 --seed 7 [--life]
```

1. Sample 30 moments from your 1:1 chats where they texted and you replied.
2. For each, the copy gets the chat up to that moment, the person's note, and your profile. Your real reply is
   **hidden and removed from its examples** (otherwise it finds its own answer among similar messages). Lines in
   the note that quote the hidden reply are dropped too.
   - `--life` also gives it your messages in other chats in the two days *before* the moment, never after.
3. A judge model (`JUDGE_CHAIN`) sees 15 random messages of yours for reference, the last few messages of the chat,
   and both replies in random order. It has to say which one is real and what gave the other away.
4. **Fooled = it picked the copy.** 15/30 (50%) means it's guessing.

The report (`reports/eval_*.md`) shows the setup, your numbers against the copy's (messages per reply, words,
questions, emoji, swearing, seconds per reply), every "tell", and every moment side by side.

**Noise.** With 30 moments, the same setup scores ±2-3 between runs. Use the same `--seed` to compare two setups,
and trust only differences bigger than that, or run 60.

## What moved the score

Measured on one person's chats: ~16k replies, Hinglish, mostly short. Same 30 moments (seed 7), same judge
(Opus, low effort). Your numbers will differ. The order of what helps probably won't.

| Setup | Fooled | What changed |
|---|---|---|
| One big prompt, one reply | 5/30 | baseline: profile + similar examples |
| Cached prompt + lookup tools + name masking | 6/30 | cheaper and safer; about the same quality |
| + your days (your lines in other chats around then) | 10/30 | the copy can bring up its own life, like real people do |
| + 4 candidates, judge-style picker held to your length mix | 12/30 | the picker alone drifted to one-word replies; the length limit fixed it |
| + option C: your fine-tuned 8B rewords every candidate | **15/30** | your exact words, spelling and stretching ("yesss") |

What didn't help, or made it worse:

- **Best-of-4 from a cheaper model.** More options from a weaker writer didn't beat 4 from a strong one.
- **A free judge-style picker.** One-word replies 83-90% of the time against a real 63%. It fooled the judge more
  but made the copy dull, so it isn't worth it.
- **More rules about style.** Every "be casual, use slang" rule made replies more uniform, and uniform is what the
  judge catches. Real examples beat descriptions.
- **Pasting the note's real exchanges.** The model copies them word for word. They're kept out of the prompt.

## What still gives it away

From the judge's "tells" at 15/30:

- **Safe fillers** ("same to you", "bolo", "ok"). Real replies are more specific and more random.
- **Ignoring a reel or photo** it can't see. A real reply reacts to what's in it.
- **Too short too often** (77% one-message replies against a real 63%).
- **Missing specifics only you know**: a name, a place, last week's plan. `me/now.md` and good notes help most here.

## Tips

- Run the test before and after *every* change to prompts, filters or the picker, with the same seed.
- Read the side-by-side, not just the score. Ten tells usually point at one fix.
- A held-out test (moments the fine-tuned model never saw) matters for option C. `finetune dataset` holds out
  the last 14 days of every chat, so use moments from there to be fair.
