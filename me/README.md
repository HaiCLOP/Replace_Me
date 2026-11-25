# `me/`: your data (stays on your machine)

Everything about you lives here. Git ignores it all except these templates, so you can't push your chats by accident.

| File / folder | What it is | Who writes it |
|---|---|---|
| `export/` | Your Instagram chat export (JSON). See `export/README.md`. | you (download from Instagram) |
| `profile.yaml` | Who you are, the people in your life, their closeness score, rules. Start from `profile.example.yaml`. | you (10 min) |
| `now.md` | What's going on in your life right now: what you're building, playing, studying, this week's plans. Read before every reply. | you, whenever things change |
| `people/<chat>.md` | Notes on how you text with each person (a strong model reads each chat once). | `python -m replace_me research people` |
| `people/_me.md` | Your master profile: how you text with everyone. | `python -m replace_me research me` |
| `mask_extra.txt` | Extra names to hide from cloud models (one per line). | you, optional |
| `mask_keep.txt` | Words that look like names but must never be hidden. | you, optional |

`python -m replace_me init` copies the `*.example.*` templates to their real names.

Edit the notes in `people/` freely: they're plain markdown, and fixing a wrong guess there is the fastest way to
make the copy better with one person.
