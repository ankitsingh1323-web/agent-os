"""Agent base class.

An agent is a role + a bound model + a capability set. It does its work purely
through the :class:`AgentContext` syscall surface, so agents are portable across
deployments and never assume anything about the machine they run on beyond what
the OS grants them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentResult:
    output: str
    success: bool = True
    meta: dict = field(default_factory=dict)


class Agent:
    #: logical model name (resolved by the ModelRegistry)
    model: str = "qwen"
    #: short role name used for routing and stub responses
    role: str = "assistant"
    #: one-line description used by the router
    description: str = "A general-purpose agent."
    #: capability tags the router can match against
    capabilities: List[str] = []
    #: tools this agent is allowed to call
    allowed_tools: List[str] = []

    def __init__(self, name: Optional[str] = None, model: Optional[str] = None) -> None:
        self.name = name or self.role
        if model:
            self.model = model

    def system_prompt(self) -> str:
        """System prompt. The ``[ROLE: ...]`` marker lets the offline stub
        backend respond in-character; real models simply read it as context."""
        return f"[ROLE: {self.role}] You are the {self.role} agent in an airgapped Agent OS."

    def build_messages(self, ctx) -> list:
        """Assemble the chat transcript for this task. Override to customize.

        Results from dependency tasks are threaded in automatically so each agent
        sees what upstream agents produced.
        """
        messages = [{"role": "system", "content": self.system_prompt()}]
        upstream = ctx.task.payload.get("upstream") or {}
        if upstream:
            joined = "\n\n".join(f"[{k}]\n{v}" for k, v in upstream.items())
            messages.append(
                {"role": "user", "content": f"Context from prior steps:\n{joined}"}
            )
        messages.append({"role": "user", "content": ctx.task.goal})
        return messages

    async def act(self, ctx) -> AgentResult:
        """Default behavior: one inference over the built messages."""
        output = await ctx.infer(self.build_messages(ctx))
        return AgentResult(output=output)
