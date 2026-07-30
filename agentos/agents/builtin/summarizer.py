"""Summarizer — synthesizes all sub-task results into a final answer."""

from __future__ import annotations

from agentos.agents.base import Agent, AgentResult


class SummarizerAgent(Agent):
    model = "kimi"
    role = "summarizer"
    description = "Synthesizes all step results into a single coherent answer."
    capabilities = ["synthesis", "summarization"]

    def system_prompt(self) -> str:
        return (
            "[ROLE: summarizer] You are the summarizer in an airgapped Agent OS. "
            "Integrate the results of all prior steps into one clear, actionable "
            "answer to the original goal. Be concise and decisive."
        )
