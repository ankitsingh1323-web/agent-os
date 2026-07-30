"""Per-task working memory — the short-lived scratchpad handed to an agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class WorkingMemory:
    task_id: str
    scratch: List[str] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)

    def note(self, text: str) -> None:
        self.scratch.append(text)

    def put(self, key: str, value: object) -> None:
        self.artifacts[key] = value

    def get(self, key: str, default: object = None) -> object:
        return self.artifacts.get(key, default)
