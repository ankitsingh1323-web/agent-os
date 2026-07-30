"""Shared blackboard — the common workspace agents read and write.

A classic blackboard architecture: agents post partial results others can pick
up. Persisted to a local JSON file so a run survives a restart (airgap-friendly,
no database required).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional


class Blackboard:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._data: dict = {}
        self._log: list = []
        if self._path and self._path.exists():
            loaded = json.loads(self._path.read_text())
            self._data = loaded.get("data", {})
            self._log = loaded.get("log", [])

    def set(self, key: str, value: object, by: str = "system") -> None:
        with self._lock:
            self._data[key] = value
            self._log.append({"op": "set", "key": key, "by": by})
            self._flush()

    def get(self, key: str, default: object = None) -> object:
        with self._lock:
            return self._data.get(key, default)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def _flush(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"data": self._data, "log": self._log}, indent=2))
