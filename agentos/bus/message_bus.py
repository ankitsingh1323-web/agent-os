"""In-process async pub/sub bus.

Agents don't call each other directly; they publish to topics and the kernel or
peer agents subscribe. This keeps coupling loose and gives us one place to audit
every message that crosses between agents.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List

Handler = Callable[["BusMessage"], Awaitable[None]]


@dataclass
class BusMessage:
    topic: str
    sender: str
    payload: object
    seq: int = 0
    meta: dict = field(default_factory=dict)


class MessageBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Handler]] = {}
        self._history: List[BusMessage] = []
        self._seq = 0

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs.setdefault(topic, []).append(handler)

    async def publish(self, topic: str, sender: str, payload: object, **meta) -> BusMessage:
        self._seq += 1
        msg = BusMessage(topic=topic, sender=sender, payload=payload, seq=self._seq, meta=meta)
        self._history.append(msg)
        handlers = list(self._subs.get(topic, [])) + list(self._subs.get("*", []))
        if handlers:
            await asyncio.gather(*(h(msg) for h in handlers))
        return msg

    def history(self, topic: str | None = None) -> List[BusMessage]:
        if topic is None:
            return list(self._history)
        return [m for m in self._history if m.topic == topic]
