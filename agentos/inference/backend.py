"""Model backends.

A ``Backend`` turns a list of chat messages into a completion string. Three
implementations ship:

* ``LocalOpenAIBackend`` — talks the OpenAI-compatible ``/chat/completions`` API
  that Ollama, vLLM, llama.cpp-server and LM Studio all expose on localhost.
* ``StubBackend`` — a deterministic, weight-free responder so the whole OS runs
  and tests pass inside an airgap with no model loaded.
* ``AutoBackend`` — probes a local endpoint once; uses it if reachable, else
  transparently falls back to the stub. This is the default so the demo "just
  works" whether or not you have models running.

Only the standard library is used, so the core installs nothing.
"""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse

from agentos.inference.airgap import AirgapPolicy, assert_local

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass
class GenerationParams:
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95
    stop: List[str] = field(default_factory=list)

    def to_openai(self) -> dict:
        body = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.stop:
            body["stop"] = self.stop
        return body


class Backend:
    """Abstract model backend."""

    name = "backend"

    async def generate(
        self, messages: List[Message], params: GenerationParams, model_id: str
    ) -> str:
        raise NotImplementedError

    async def healthy(self) -> bool:
        return True


class LocalOpenAIBackend(Backend):
    """OpenAI-compatible HTTP client, pinned to a local endpoint.

    Works with any runtime that serves ``POST {base_url}/chat/completions``.
    """

    name = "local-openai"

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "local",
        timeout: float = 120.0,
        policy: AirgapPolicy | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.policy = policy
        assert_local(self.base_url, policy)  # fail fast on misconfiguration

    def _request(self, body: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        assert_local(url, self.policy)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        # No proxies: an airgapped box should never route through one.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def generate(
        self, messages: List[Message], params: GenerationParams, model_id: str
    ) -> str:
        body = {"model": model_id, "messages": messages, **params.to_openai()}
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(None, self._request, body)
        return payload["choices"][0]["message"]["content"]

    async def healthy(self) -> bool:
        host = urlparse(self.base_url).hostname or "localhost"
        port = urlparse(self.base_url).port or 80
        loop = asyncio.get_event_loop()

        def _probe() -> bool:
            try:
                with socket.create_connection((host, port), timeout=1.5):
                    return True
            except OSError:
                return False

        return await loop.run_in_executor(None, _probe)


class StubBackend(Backend):
    """Deterministic offline backend — no weights, no network.

    It keys off a ``[ROLE: <name>]`` marker that agents place in their system
    prompt, so the demo pipeline produces coherent, role-appropriate output and
    tests stay deterministic.
    """

    name = "stub"

    async def generate(
        self, messages: List[Message], params: GenerationParams, model_id: str
    ) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        role = "assistant"
        if "[ROLE:" in system:
            role = system.split("[ROLE:", 1)[1].split("]", 1)[0].strip().lower()
        goal = user.strip().replace("\n", " ")
        goal_short = (goal[:180] + "…") if len(goal) > 180 else goal
        return _STUB_ROLES.get(role, _stub_generic)(goal_short, model_id)


def _stub_generic(goal: str, model: str) -> str:
    return f"[{model}·stub] Response to: {goal}"


def _stub_planner(goal: str, model: str) -> str:
    plan = {
        "steps": [
            {
                "id": "s1",
                "agent": "researcher",
                "goal": f"Gather the key facts, constraints and prior art for: {goal}",
                "deps": [],
            },
            {
                "id": "s2",
                "agent": "coder",
                "goal": f"Produce a concrete solution/draft addressing: {goal}",
                "deps": ["s1"],
            },
            {
                "id": "s3",
                "agent": "reviewer",
                "goal": "Review the draft for correctness, risks and gaps; suggest fixes.",
                "deps": ["s2"],
            },
        ]
    }
    return json.dumps(plan, indent=2)


def _stub_researcher(goal: str, model: str) -> str:
    return (
        f"Research notes ({model}·stub):\n"
        f"- Objective: {goal}\n"
        "- Relevant considerations: correctness, offline/airgap constraints, "
        "resource limits of local open-weight models.\n"
        "- Prior art: standard orchestration patterns (plan → act → review)."
    )


def _stub_coder(goal: str, model: str) -> str:
    return (
        f"Proposed solution ({model}·stub) for: {goal}\n"
        "1. Decompose the problem into small, testable units.\n"
        "2. Implement the core path first, guard the edges.\n"
        "3. Add a smoke test and a rollback path."
    )


def _stub_reviewer(goal: str, model: str) -> str:
    return (
        f"Review ({model}·stub):\n"
        "- Strengths: clear structure, addresses the stated goal.\n"
        "- Risks: verify edge cases and failure handling.\n"
        "- Verdict: APPROVE with minor changes."
    )


def _stub_summarizer(goal: str, model: str) -> str:
    return (
        f"Summary ({model}·stub): The team researched, drafted, and reviewed a "
        f"solution for '{goal}'. Outcome: a coherent plan validated end-to-end."
    )


_STUB_ROLES = {
    "planner": _stub_planner,
    "researcher": _stub_researcher,
    "coder": _stub_coder,
    "reviewer": _stub_reviewer,
    "critic": _stub_reviewer,
    "summarizer": _stub_summarizer,
}


class AutoBackend(Backend):
    """Use a local runtime if it answers, otherwise fall back to the stub.

    The reachability probe runs once and is cached, so an airgapped machine with
    no model server still runs the full OS deterministically.
    """

    name = "auto"

    def __init__(self, primary: Backend, fallback: Backend | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or StubBackend()
        self._use_primary: bool | None = None

    async def _resolve(self) -> Backend:
        if self._use_primary is None:
            self._use_primary = await self.primary.healthy()
        return self.primary if self._use_primary else self.fallback

    async def generate(
        self, messages: List[Message], params: GenerationParams, model_id: str
    ) -> str:
        backend = await self._resolve()
        try:
            return await backend.generate(messages, params, model_id)
        except (urllib.error.URLError, OSError):
            # Runtime went away mid-run — degrade gracefully to the stub.
            self._use_primary = False
            return await self.fallback.generate(messages, params, model_id)
