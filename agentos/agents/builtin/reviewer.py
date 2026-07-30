"""Reviewer — checks a draft for correctness, risks and gaps."""

from __future__ import annotations

from agentos.agents.base import Agent, AgentResult


class ReviewerAgent(Agent):
    model = "llama"
    role = "reviewer"
    description = "Reviews a draft for correctness, risk and completeness."
    capabilities = ["review", "qa", "verification"]

    def system_prompt(self) -> str:
        return (
            "[ROLE: reviewer] You are the reviewer in an airgapped Agent OS. "
            "Assess the prior draft. List strengths, concrete risks, and a verdict "
            "of APPROVE / APPROVE-WITH-CHANGES / REJECT with reasons."
        )
