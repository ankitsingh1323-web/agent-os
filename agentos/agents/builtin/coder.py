"""Coder — turns a goal + research into a concrete solution or draft."""

from __future__ import annotations

from agentos.agents.base import Agent, AgentResult


class CoderAgent(Agent):
    model = "qwen"
    role = "coder"
    description = "Produces concrete solutions, code or drafts from research."
    capabilities = ["coding", "implementation", "drafting"]
    allowed_tools = ["read_file", "write_file", "list_dir", "calculator"]

    def system_prompt(self) -> str:
        return (
            "[ROLE: coder] You are the coder in an airgapped Agent OS. "
            "Turn the goal and prior research into a concrete, minimal, correct "
            "solution. Prefer small testable units and note any assumptions."
        )

    async def act(self, ctx) -> AgentResult:
        output = await ctx.infer(self.build_messages(ctx))
        ctx.remember(output, kind="draft", tags=["draft"])
        return AgentResult(output=output)
