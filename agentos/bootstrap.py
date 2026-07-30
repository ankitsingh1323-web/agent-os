"""One-call construction of a working, airgapped Agent OS.

``build_default_os()`` wires every subsystem with sensible defaults so you can go
from import to a running orchestration in three lines — no config files, no
network, no model server required (it degrades to the offline stub).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from agentos.agents.registry import AgentRegistry, build_default_agents
from agentos.inference.airgap import AirgapPolicy
from agentos.inference.registry import ModelRegistry, build_default_registry
from agentos.kernel.kernel import Kernel
from agentos.kernel.orchestrator import Orchestrator, RunReport
from agentos.memory.blackboard import Blackboard
from agentos.memory.store import MemoryStore
from agentos.telemetry.audit import AuditLog
from agentos.tools.registry import ToolRegistry, build_default_tools


@dataclass
class AgentOS:
    """A ready-to-run instance: kernel + default orchestration policy."""

    kernel: Kernel
    orchestrator: Orchestrator

    async def run(self, goal: str) -> RunReport:
        return await self.orchestrator.run(goal)

    def describe(self) -> dict:
        return {
            "version": __import__("agentos").__version__,
            "airgap": self.kernel.models.policy.enforce if self.kernel.models.policy else True,
            "models": self.kernel.models.names(),
            "agents": self.kernel.agents.describe(),
            "tools": self.kernel.tools.describe(),
        }


def build_default_os(
    *,
    workspace: str = ".",
    state_dir: Optional[str] = None,
    model_overrides: Optional[Dict[str, str]] = None,
    policy: Optional[AirgapPolicy] = None,
    concurrency: int = 4,
    config_path: Optional[str] = None,
) -> AgentOS:
    # Optional JSON config overrides any of the code defaults below.
    if config_path:
        from agentos import config as _cfg

        cfg = _cfg.load(config_path)
        policy = policy or _cfg.policy_from(cfg)
        concurrency = cfg.get("concurrency", concurrency)
        model_overrides = {**_cfg.agent_models_from(cfg), **(model_overrides or {})}
        _extra_profiles = _cfg.profiles_from(cfg)
    else:
        _extra_profiles = []

    policy = policy or AirgapPolicy()  # strict airgap by default
    models: ModelRegistry = build_default_registry(policy=policy)
    for profile in _extra_profiles:  # config profiles win over defaults
        models.register(profile)
    agents: AgentRegistry = build_default_agents(model_overrides=model_overrides)
    tools: ToolRegistry = build_default_tools(workspace=workspace)

    def _p(name: str) -> Optional[str]:
        return f"{state_dir.rstrip('/')}/{name}" if state_dir else None

    kernel = Kernel(
        models=models,
        agents=agents,
        tools=tools,
        memory=MemoryStore(_p("memory.jsonl")),
        blackboard=Blackboard(_p("blackboard.json")),
        audit=AuditLog(_p("audit.jsonl")),
        concurrency=concurrency,
    )
    return AgentOS(kernel=kernel, orchestrator=Orchestrator(kernel))
