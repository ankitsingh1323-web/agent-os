"""Append-only audit trail.

In an airgapped system the audit log is the accountability layer: every
inference, tool call and agent spawn is recorded locally. Nothing is shipped
off-box.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AuditEvent:
    seq: int
    kind: str            # "infer" | "tool" | "spawn" | "task" | "emit"
    actor: str
    detail: dict = field(default_factory=dict)


class AuditLog:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path else None
        self._events: List[AuditEvent] = []
        self._seq = 0

    def record(self, kind: str, actor: str, **detail) -> AuditEvent:
        self._seq += 1
        event = AuditEvent(seq=self._seq, kind=kind, actor=actor, detail=detail)
        self._events.append(event)
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as fh:
                fh.write(json.dumps(asdict(event)) + "\n")
        return event

    def events(self) -> List[AuditEvent]:
        return list(self._events)
