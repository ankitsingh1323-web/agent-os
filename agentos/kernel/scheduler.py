"""Scheduler — runs tasks respecting dependencies, with bounded concurrency.

A dependency-aware wave scheduler: on each pass it finds every task whose deps
are satisfied and runs them concurrently (up to ``concurrency``), threading each
finished task's result into its dependents' payloads. It loops until everything
is terminal, and refuses to spin forever on an unsatisfiable dependency graph.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List

from agentos.kernel.task import Task, TaskStatus

Runner = Callable[[Task], Awaitable[None]]


class Scheduler:
    def __init__(self, concurrency: int = 4) -> None:
        self.concurrency = concurrency
        self._tasks: Dict[str, Task] = {}

    def add(self, task: Task) -> None:
        self._tasks[task.id] = task

    def tasks(self) -> List[Task]:
        return list(self._tasks.values())

    def _ready(self) -> List[Task]:
        ready = []
        for t in self._tasks.values():
            if t.status != TaskStatus.PENDING:
                continue
            if all(self._tasks[d].status == TaskStatus.DONE for d in t.deps if d in self._tasks):
                ready.append(t)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def _thread_upstream(self, task: Task) -> None:
        upstream = {
            d: self._tasks[d].result
            for d in task.deps
            if d in self._tasks and self._tasks[d].result is not None
        }
        if upstream:
            task.payload["upstream"] = upstream

    async def run(self, runner: Runner) -> List[Task]:
        sem = asyncio.Semaphore(self.concurrency)

        async def _run_one(task: Task) -> None:
            async with sem:
                task.status = TaskStatus.RUNNING
                self._thread_upstream(task)
                await runner(task)

        while True:
            ready = self._ready()
            if not ready:
                pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
                if not pending:
                    break  # all terminal
                # Nothing can run but tasks remain → broken/failed dependency.
                for t in pending:
                    t.status = TaskStatus.FAILED
                    t.error = "blocked: dependency failed or missing"
                break
            await asyncio.gather(*(_run_one(t) for t in ready))
        return self.tasks()
