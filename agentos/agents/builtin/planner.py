"""Planner — decomposes a goal into a dependency-ordered plan of sub-tasks."""

from __future__ import annotations

import json
import re
from typing import List

from agentos.agents.base import Agent, AgentResult


class PlannerAgent(Agent):
    model = "kimi"
    role = "planner"
    description = "Breaks a high-level goal into an ordered plan of agent sub-tasks."
    capabilities = ["planning", "decomposition", "orchestration"]

    def system_prompt(self) -> str:
        return (
            "[ROLE: planner] You are the planner in an airgapped Agent OS. "
            "Decompose the user's goal into 2-5 sub-tasks. Respond ONLY with JSON: "
            '{"steps":[{"id":"s1","agent":"researcher","goal":"...","deps":[]}]}. '
            "Valid agents: researcher, coder, reviewer, critic, summarizer."
        )

    async def act(self, ctx) -> AgentResult:
        raw = await ctx.infer(self.build_messages(ctx))
        steps = _parse_plan(raw)
        ctx.board_set("plan", steps)
        return AgentResult(output=raw, meta={"steps": steps})


def _parse_plan(raw: str) -> List[dict]:
    """Best-effort extraction of a step list from a model response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            steps = json.loads(match.group(0)).get("steps", [])
            if isinstance(steps, list) and steps:
                return [_normalize(s, i) for i, s in enumerate(steps)]
        except json.JSONDecodeError:
            pass
    # Fallback: a single research step so the OS always makes progress.
    return [{"id": "s1", "agent": "researcher", "goal": raw.strip()[:400], "deps": []}]


def _normalize(step: dict, i: int) -> dict:
    return {
        "id": step.get("id") or f"s{i + 1}",
        "agent": step.get("agent", "researcher"),
        "goal": step.get("goal", ""),
        "deps": step.get("deps", []) or [],
    }
