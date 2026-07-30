"""The syscall surface handed to an agent while it runs.

Agents never touch subsystems directly. They go through this context, which is
the OS boundary: it brokers inference, tool use, memory, messaging and spawning
of sub-agents, and audits every call. This is what makes the collection of
agents an *operating system* rather than a pile of scripts.
"""

from __future__ import annotations

from typing import List, Optional

from agentos.inference.backend import GenerationParams, Message
from agentos.memory.store import MemoryRecord
from agentos.memory.working import WorkingMemory
from agentos.tools.registry import ToolResult


class AgentContext:
    def __init__(self, kernel, task, agent) -> None:
        self._kernel = kernel
        self.task = task
        self.agent = agent
        self.working = WorkingMemory(task_id=task.id)

    # --- inference ---------------------------------------------------------
    async def infer(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        params: Optional[GenerationParams] = None,
    ) -> str:
        model = model or self.agent.model
        self._kernel.audit.record("infer", self.agent.name, model=model, task=self.task.id)
        return await self._kernel.infer(model, messages, params)

    # --- messaging ---------------------------------------------------------
    async def emit(self, topic: str, payload: object) -> None:
        self._kernel.audit.record("emit", self.agent.name, topic=topic)
        await self._kernel.bus.publish(topic, sender=self.agent.name, payload=payload)

    # --- memory ------------------------------------------------------------
    def remember(self, text: str, kind: str = "result", tags: Optional[List[str]] = None) -> None:
        rec = MemoryRecord(
            id=f"{self.task.id}:{len(self._kernel.memory.all())}",
            kind=kind,
            text=text,
            source=self.agent.name,
            tags=tags or [],
        )
        self._kernel.memory.add(rec)

    def recall(self, query: str, k: int = 5) -> List[MemoryRecord]:
        return self._kernel.memory.search(query, k)

    def board_get(self, key: str, default: object = None) -> object:
        return self._kernel.blackboard.get(key, default)

    def board_set(self, key: str, value: object) -> None:
        self._kernel.blackboard.set(key, value, by=self.agent.name)

    # --- tools -------------------------------------------------------------
    async def use_tool(self, name: str, **kwargs) -> ToolResult:
        self._kernel.audit.record("tool", self.agent.name, tool=name)
        return self._kernel.run_tool(self.agent, name, **kwargs)

    # --- sub-agents --------------------------------------------------------
    async def spawn(self, agent: str, goal: str, wait: bool = True) -> Optional[str]:
        self._kernel.audit.record("spawn", self.agent.name, child=agent, goal=goal)
        return await self._kernel.spawn(agent, goal, parent=self.task.id, wait=wait)

    def log(self, message: str) -> None:
        self.working.note(message)
