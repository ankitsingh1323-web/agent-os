"""The unit of scheduled work."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional


class TaskStatus(str, enum.Enum):
    PENDING = "pending"      # created, deps not yet satisfied
    READY = "ready"          # deps satisfied, waiting for a worker
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    goal: str
    agent: Optional[str] = None          # assigned agent name; None → router decides
    deps: List[str] = field(default_factory=list)
    priority: int = 0                    # higher runs first among ready tasks
    payload: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
