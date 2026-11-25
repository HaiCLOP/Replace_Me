"""Small structured logger: one JSON line per event in logs/replace_me.jsonl, a short line on the console."""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from replace_me.config import ROOT

_LOG = ROOT / "logs" / "replace_me.jsonl"
_lock = threading.Lock()


class Logger:
    def __init__(self, name: str):
        self.name = name

    def _write(self, level: str, event: str, fields: dict[str, Any]) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "logger": self.name, "event": event, **fields}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with _lock:
            _LOG.parent.mkdir(parents=True, exist_ok=True)
            with _LOG.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        if level != "debug":
            short = " ".join(f"{k}={str(v)[:80]}" for k, v in fields.items())
            print(f"[{level}] {event} {short}", file=sys.stderr if level in ("warning", "error") else sys.stdout,
                  flush=True)

    def debug(self, event: str, **kw: Any) -> None:
        self._write("debug", event, kw)

    def info(self, event: str, **kw: Any) -> None:
        self._write("info", event, kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._write("warning", event, kw)

    def error(self, event: str, **kw: Any) -> None:
        self._write("error", event, kw)


def get_logger(name: str) -> Logger:
    return Logger(name)


def log_path() -> Path:
    return _LOG
