"""Critic — adversarial check; tries to find where the plan or draft fails."""

from __future__ import annotations

from agentos.agents.base import Agent, AgentResult


class CriticAgent(Agent):
    model = "llama"
    role = "critic"
    description = "Adversarially probes a result for failure modes and blind spots."
    capabilities = ["critique", "red-team", "verification"]

    def system_prompt(self) -> str:
        return (
            "[ROLE: critic] You are the critic in an airgapped Agent OS. "
            "Adversarially probe the prior work: what breaks it, what did it miss, "
            "which assumptions are unsafe? Be specific and constructive."
        )
