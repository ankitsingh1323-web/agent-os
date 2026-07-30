"""Adapter for externally-built agents.

The whole point of the OS is a *narrow waist*: a thin, mandatory contract at the
boundary, total freedom behind it. This module is how an agent built any other
way — LangChain, AutoGen, CrewAI, a raw prompt loop, even a human-in-a-function —
runs under the kernel without special-casing.

Two pieces:

* :class:`AgentHarness` — the framework-agnostic "syscall SDK" handed to foreign
  code. It is the *only* sanctioned way to reach a model, a tool, or memory, and
  every call is routed through the kernel (so the airgap, audit trail and
  permission model all still hold).
* :class:`ExternalAgent` — wraps a foreign ``runner`` as a first-class kernel
  agent. To integrate a framework you write a small runner that calls
  ``harness.infer(...)`` instead of its own LLM client — that is the entire
  integration surface.

The iron rule of an airgapped OS — *no agent performs its own I/O* — is not left
to good behavior: the runner executes inside :func:`process_egress_guard`, so any
attempt at non-local network access is refused at the socket.
"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from typing import Awaitable, Callable, List, Optional, Union

from agentos.agents.base import Agent, AgentResult
from agentos.inference.airgap import process_egress_guard
from agentos.inference.backend import GenerationParams, Message
from agentos.memory.store import MemoryRecord
from agentos.tools.registry import ToolResult

# A runner is any callable that takes a harness and returns a result. It may be
# sync or async, and return a string, a dict, or an AgentResult.
Runner = Callable[["AgentHarness"], Union[str, dict, AgentResult, Awaitable]]


class AgentHarness:
    """The stable, framework-agnostic surface foreign agents build against.

    Everything here forwards to a kernel syscall — foreign code never touches a
    subsystem, a socket, or a model client directly.
    """

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    # --- what the task is --------------------------------------------------
    @property
    def goal(self) -> str:
        return self._ctx.task.goal

    @property
    def upstream(self) -> dict:
        """Results of dependency tasks, keyed by step id."""
        return dict(self._ctx.task.payload.get("upstream") or {})

    @property
    def model(self) -> str:
        return self._ctx.agent.model

    # --- inference (the sanctioned model path) -----------------------------
    async def infer(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        params: Optional[GenerationParams] = None,
    ) -> str:
        """One-shot completion from a plain prompt. Goes through the airgap guard
        and the audit log like every other model call."""
        messages: List[Message] = [
            {"role": "system", "content": system or self._ctx.agent.system_prompt()},
            {"role": "user", "content": prompt},
        ]
        return await self._ctx.infer(messages, model=model, params=params)

    async def chat(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        params: Optional[GenerationParams] = None,
    ) -> str:
        """Completion from a full message list, for agents that manage their own
        transcript."""
        return await self._ctx.infer(messages, model=model, params=params)

    # --- tools / memory / messaging ---------------------------------------
    async def use_tool(self, name: str, **kwargs) -> ToolResult:
        return await self._ctx.use_tool(name, **kwargs)

    def remember(self, text: str, kind: str = "result", tags: Optional[List[str]] = None) -> None:
        self._ctx.remember(text, kind=kind, tags=tags)

    def recall(self, query: str, k: int = 5) -> List[MemoryRecord]:
        return self._ctx.recall(query, k)

    def board_get(self, key: str, default: object = None) -> object:
        return self._ctx.board_get(key, default)

    def board_set(self, key: str, value: object) -> None:
        self._ctx.board_set(key, value)

    async def emit(self, topic: str, payload: object) -> None:
        await self._ctx.emit(topic, payload)

    def log(self, message: str) -> None:
        self._ctx.log(message)


class ExternalAgent(Agent):
    """Wrap a foreign ``runner`` as a first-class kernel agent.

    It declares the four things the contract requires — identity, capabilities,
    a single entry point, and structured output — and satisfies the fifth
    invariant (no self-I/O) by running the foreign code under the egress guard.
    """

    def __init__(
        self,
        name: str,
        runner: Runner,
        *,
        model: str = "qwen",
        role: str = "external",
        description: str = "An externally-built agent adapted to the kernel contract.",
        capabilities: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        enforce_egress: bool = True,
    ) -> None:
        super().__init__(name=name, model=model)
        self._runner = runner
        self.role = role
        self.description = description
        self.capabilities = capabilities or []
        self.allowed_tools = allowed_tools or []
        self.enforce_egress = enforce_egress

    async def act(self, ctx) -> AgentResult:
        harness = AgentHarness(ctx)
        guard = process_egress_guard(ctx.airgap_policy) if self.enforce_egress else nullcontext()
        with guard:
            result = self._runner(harness)
            if inspect.isawaitable(result):
                result = await result
        return _normalize(result)


def adapt(name: str, runner: Runner, **kwargs) -> ExternalAgent:
    """Convenience factory: ``registry.register(adapt("my-agent", my_runner))``."""
    return ExternalAgent(name, runner, **kwargs)


def _normalize(result: Union[str, dict, AgentResult, None]) -> AgentResult:
    if isinstance(result, AgentResult):
        return result
    if isinstance(result, dict):
        return AgentResult(
            output=str(result.get("output", "")),
            success=bool(result.get("success", True)),
            meta=dict(result.get("meta", {})),
        )
    if result is None:
        return AgentResult(output="", success=False, meta={"reason": "runner returned None"})
    return AgentResult(output=str(result))
