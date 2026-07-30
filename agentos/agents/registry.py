"""Agent registry — the catalog the kernel routes tasks against."""

from __future__ import annotations

from typing import Dict, List

from agentos.agents.base import Agent
from agentos.agents.builtin import ALL_AGENTS


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        if name not in self._agents:
            raise KeyError(f"Unknown agent '{name}'. Known: {sorted(self._agents)}")
        return self._agents[name]

    def has(self, name: str) -> bool:
        return name in self._agents

    def all(self) -> List[Agent]:
        return list(self._agents.values())

    def names(self) -> List[str]:
        return sorted(self._agents)

    def describe(self) -> List[dict]:
        return [
            {
                "name": a.name,
                "role": a.role,
                "model": a.model,
                "description": a.description,
                "capabilities": a.capabilities,
            }
            for a in self._agents.values()
        ]


def build_default_agents(model_overrides: Dict[str, str] | None = None) -> AgentRegistry:
    """Instantiate the built-in fleet. ``model_overrides`` re-binds an agent to a
    different logical model, e.g. ``{"coder": "llama"}``."""
    overrides = model_overrides or {}
    reg = AgentRegistry()
    for cls in ALL_AGENTS:
        agent = cls()
        if agent.name in overrides:
            agent.model = overrides[agent.name]
        reg.register(agent)
    return reg
