"""Researcher — gathers facts and context (from local memory only)."""

from __future__ import annotations

from agentos.agents.base import Agent, AgentResult


class ResearcherAgent(Agent):
    model = "qwen"
    role = "researcher"
    description = "Gathers relevant facts, constraints and prior art for a task."
    capabilities = ["research", "analysis", "retrieval"]
    allowed_tools = ["read_file", "list_dir"]

    def system_prompt(self) -> str:
        return (
            "[ROLE: researcher] You are the researcher in an airgapped Agent OS. "
            "Produce concise, well-organized findings for the given goal using only "
            "local knowledge and the provided context. Cite assumptions explicitly."
        )

    async def act(self, ctx) -> AgentResult:
        recalled = ctx.recall(ctx.task.goal, k=3)
        if recalled:
            note = "Relevant prior memory:\n" + "\n".join(f"- {r.text}" for r in recalled)
            ctx.log(note)
        output = await ctx.infer(self.build_messages(ctx))
        ctx.remember(output, kind="research", tags=["research"])
        return AgentResult(output=output)
