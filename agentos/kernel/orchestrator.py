"""Orchestrator — the top-level "shell" that turns a goal into a finished answer.

Flow:
    1. Planner decomposes the goal into a dependency-ordered plan.
    2. Each step becomes a scheduled Task assigned to an agent.
    3. The scheduler runs them, threading results along dependencies.
    4. The summarizer synthesizes everything into the final answer.

This is a sensible default policy, not the only one — swap it for a debate loop,
a market, or a fixed pipeline by writing another orchestrator over the same
kernel syscalls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from agentos.kernel.kernel import Kernel
from agentos.kernel.task import Task, TaskStatus


@dataclass
class RunReport:
    goal: str
    plan: List[dict] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)
    answer: str = ""

    def trace(self) -> str:
        lines = [f"GOAL: {self.goal}", ""]
        for t in self.tasks:
            head = f"[{t.status.value.upper():7}] {t.id} · {t.agent}"
            body = (t.result or t.error or "").strip().splitlines()
            lines.append(head)
            lines.extend("    " + ln for ln in body[:6])
            lines.append("")
        lines.append("FINAL ANSWER:")
        lines.append(self.answer.strip())
        return "\n".join(lines)


class Orchestrator:
    def __init__(self, kernel: Kernel, planner: str = "planner", summarizer: str = "summarizer") -> None:
        self.kernel = kernel
        self.planner = planner
        self.summarizer = summarizer

    async def run(self, goal: str) -> RunReport:
        report = RunReport(goal=goal)

        # 1. Plan
        plan_task = self.kernel.new_task(goal, agent=self.planner)
        await self.kernel.run_task(plan_task)
        steps = self.kernel.blackboard.get("plan") or plan_task.payload.get("steps") or []
        report.plan = steps

        # 2. Turn the plan into scheduled tasks (preserving step ids for deps)
        id_map = {}
        step_tasks: List[Task] = []
        for step in steps:
            task = self.kernel.new_task(step["goal"], agent=step["agent"])
            id_map[step["id"]] = task.id
            task.payload["step_id"] = step["id"]
            task.payload["_raw_deps"] = step.get("deps", [])
            step_tasks.append(task)
        for task in step_tasks:  # remap step ids → real task ids
            task.deps = [id_map[d] for d in task.payload["_raw_deps"] if d in id_map]
            self.kernel.submit(task)

        # 3. Execute the graph
        await self.kernel.run_all()
        report.tasks = [plan_task] + step_tasks

        # 4. Synthesize
        results = {
            t.payload.get("step_id", t.id): (t.result or t.error or "")
            for t in step_tasks
            if t.status == TaskStatus.DONE
        }
        summary_task = self.kernel.new_task(
            f"Original goal: {goal}\nSynthesize the step results into a final answer.",
            agent=self.summarizer,
        )
        summary_task.payload["upstream"] = results
        await self.kernel.run_task(summary_task)
        report.tasks.append(summary_task)
        report.answer = summary_task.result or ""
        return report
