"""Router — decides which agent handles a task when none is pre-assigned.

Two strategies: an explicit assignment on the task always wins; otherwise a
lightweight capability/keyword match picks the best-fit agent. A future strategy
can delegate routing to a small local model, but the deterministic path keeps the
common case fast and offline.
"""

from __future__ import annotations

from typing import Optional

from agentos.agents.registry import AgentRegistry
from agentos.kernel.task import Task

# keyword → capability hints for the fallback matcher
_HINTS = {
    "plan": "planning",
    "research": "research",
    "find": "research",
    "code": "coding",
    "implement": "coding",
    "build": "coding",
    "write": "drafting",
    "review": "review",
    "check": "review",
    "critique": "critique",
    "risk": "critique",
    "summar": "synthesis",
}


class Router:
    def __init__(self, agents: AgentRegistry) -> None:
        self.agents = agents

    def route(self, task: Task) -> str:
        if task.agent and self.agents.has(task.agent):
            return task.agent
        return self._match(task.goal) or "researcher"

    def _match(self, goal: str) -> Optional[str]:
        goal_l = goal.lower()
        wanted = {cap for kw, cap in _HINTS.items() if kw in goal_l}
        if not wanted:
            return None
        best, best_score = None, 0
        for agent in self.agents.all():
            score = len(wanted & set(agent.capabilities))
            if score > best_score:
                best, best_score = agent.name, score
        return best
