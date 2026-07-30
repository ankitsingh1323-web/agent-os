"""The kernel — the privileged core that owns every subsystem.

Agents run in "user space": they can only reach models, tools, memory and each
other through syscalls the kernel brokers (see :class:`AgentContext`). The kernel
enforces permissions, records the audit trail, and drives task execution through
the scheduler and router.
"""

from __future__ import annotations

from typing import List, Optional

from agentos.agents.registry import AgentRegistry
from agentos.bus.message_bus import MessageBus
from agentos.inference.backend import GenerationParams, Message
from agentos.inference.registry import ModelRegistry
from agentos.kernel.context import AgentContext
from agentos.kernel.router import Router
from agentos.kernel.scheduler import Scheduler
from agentos.kernel.task import Task, TaskStatus
from agentos.memory.blackboard import Blackboard
from agentos.memory.store import MemoryStore
from agentos.telemetry.audit import AuditLog
from agentos.tools.registry import ToolRegistry, ToolResult

# Which permission tiers each capability level may invoke.
_PERMISSION_ORDER = {"safe": 0, "workspace": 1, "privileged": 2}


class Kernel:
    def __init__(
        self,
        models: ModelRegistry,
        agents: AgentRegistry,
        tools: ToolRegistry,
        *,
        memory: Optional[MemoryStore] = None,
        blackboard: Optional[Blackboard] = None,
        audit: Optional[AuditLog] = None,
        bus: Optional[MessageBus] = None,
        concurrency: int = 4,
        max_tool_permission: str = "workspace",
    ) -> None:
        self.models = models
        self.agents = agents
        self.tools = tools
        self.memory = memory or MemoryStore()
        self.blackboard = blackboard or Blackboard()
        self.audit = audit or AuditLog()
        self.bus = bus or MessageBus()
        self.router = Router(agents)
        self.scheduler = Scheduler(concurrency=concurrency)
        self.max_tool_permission = max_tool_permission
        self._counter = 0

    # --- syscalls served to agents ----------------------------------------
    async def infer(
        self, model: str, messages: List[Message], params: Optional[GenerationParams]
    ) -> str:
        backend = self.models.backend(model)
        params = params or self.models.profile(model).params
        return await backend.generate(messages, params, self.models.profile(model).model_id)

    def run_tool(self, agent, name: str, **kwargs) -> ToolResult:
        tool = self.tools.get(name)
        if name not in agent.allowed_tools:
            return ToolResult(False, None, f"agent '{agent.name}' may not call '{name}'")
        if _PERMISSION_ORDER[tool.permission] > _PERMISSION_ORDER[self.max_tool_permission]:
            return ToolResult(False, None, f"tool '{name}' exceeds kernel permission ceiling")
        return tool.run(**kwargs)

    async def spawn(self, agent_name: str, goal: str, parent: Optional[str] = None, wait: bool = True) -> Optional[str]:
        task = self.new_task(goal, agent=agent_name)
        task.payload["parent"] = parent
        await self.run_task(task)
        return task.result if wait else None

    # --- task lifecycle ----------------------------------------------------
    def new_task(self, goal: str, agent: Optional[str] = None, deps: Optional[List[str]] = None,
                 priority: int = 0) -> Task:
        self._counter += 1
        return Task(id=f"t{self._counter}", goal=goal, agent=agent, deps=deps or [], priority=priority)

    def submit(self, task: Task) -> None:
        self.scheduler.add(task)

    async def run_task(self, task: Task) -> Task:
        agent_name = self.router.route(task)
        agent = self.agents.get(agent_name)
        task.agent = agent_name
        self.audit.record("task", agent_name, task=task.id, goal=task.goal)
        ctx = AgentContext(self, task, agent)
        try:
            result = await agent.act(ctx)
            task.result = result.output
            task.status = TaskStatus.DONE if result.success else TaskStatus.FAILED
        except Exception as exc:  # never let one agent take down the kernel
            task.status = TaskStatus.FAILED
            task.error = f"{type(exc).__name__}: {exc}"
        await self.bus.publish("task.finished", sender=agent_name, payload=task)
        return task

    async def run_all(self) -> List[Task]:
        """Run every submitted task, honoring dependencies."""
        return await self.scheduler.run(self.run_task)
