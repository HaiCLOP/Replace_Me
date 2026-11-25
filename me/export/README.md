# Your Instagram export goes here

1. Instagram app or website: **Settings → Accounts Center → Your information and permissions → Download your
   information → Download or transfer information**.
2. Pick your account → **Some of your information** → tick only **Messages**.
3. **Download to device**, Date range **All time**, Format **JSON** (not HTML), Media quality **Low**.
4. Instagram emails you when it's ready (minutes to a day). Unzip it **into this folder**.

You should end up with something like `me/export/your_instagram_activity/messages/inbox/<name_12345>/message_1.json`.
Any depth works: everything named `message_*.json` under `me/export/` is read.

Then: `python -m replace_me ingest`.

Git ignores everything in this folder except this README. Your chats include other people's messages: keep them
private, and see `docs/PRIVACY.md` before sending anything to a cloud service.
