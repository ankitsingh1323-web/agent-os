"""Model registry.

Maps logical model names (``kimi``, ``qwen``, ``llama``, ``grok``) to a concrete
local runtime, model id and default sampling params. Agents reference models by
logical name, so you can re-point ``coder`` from Qwen to Llama by editing config,
never code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from agentos.inference.airgap import AirgapPolicy
from agentos.inference.backend import (
    AutoBackend,
    Backend,
    GenerationParams,
    LocalOpenAIBackend,
    StubBackend,
)


@dataclass
class ModelProfile:
    """A logical model bound to a local runtime."""

    name: str
    model_id: str
    backend: str = "auto"  # "auto" | "local-openai" | "stub"
    base_url: str = "http://localhost:11434/v1"
    context_window: int = 8192
    params: GenerationParams = field(default_factory=GenerationParams)
    notes: str = ""

    def build(self, policy: AirgapPolicy | None = None) -> Backend:
        if self.backend == "stub":
            return StubBackend()
        if self.backend == "nano":
            from agentos.inference.nanolm import NanoLMBackend
            return NanoLMBackend()
        local = LocalOpenAIBackend(base_url=self.base_url, policy=policy)
        if self.backend == "local-openai":
            return local
        return AutoBackend(primary=local)  # "auto"


class ModelRegistry:
    def __init__(self, policy: AirgapPolicy | None = None) -> None:
        self.policy = policy
        self._profiles: Dict[str, ModelProfile] = {}
        self._backends: Dict[str, Backend] = {}

    def register(self, profile: ModelProfile) -> None:
        self._profiles[profile.name] = profile
        self._backends.pop(profile.name, None)  # rebuild lazily

    def profile(self, name: str) -> ModelProfile:
        if name not in self._profiles:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self._profiles)}")
        return self._profiles[name]

    def backend(self, name: str) -> Backend:
        if name not in self._backends:
            self._backends[name] = self.profile(name).build(self.policy)
        return self._backends[name]

    def names(self) -> List[str]:
        return sorted(self._profiles)


def default_profiles() -> List[ModelProfile]:
    """The open-weight lineup this OS targets, all served locally.

    ``model_id`` values follow common Ollama tags; adjust to match your runtime.
    Backend defaults to ``auto`` so everything runs even with no server up.
    """
    return [
        ModelProfile(
            name="kimi",
            model_id="kimi-k2",
            context_window=131072,
            params=GenerationParams(temperature=0.4, max_tokens=2048),
            notes="Moonshot Kimi — long-context reasoning; used for planning/synthesis.",
        ),
        ModelProfile(
            name="qwen",
            model_id="qwen2.5:14b",
            context_window=32768,
            params=GenerationParams(temperature=0.5, max_tokens=1536),
            notes="Alibaba Qwen — strong general + coding; used for research/coding.",
        ),
        ModelProfile(
            name="llama",
            model_id="llama3.1:8b",
            context_window=16384,
            params=GenerationParams(temperature=0.6, max_tokens=1024),
            notes="Meta Llama — balanced general model; used for review/critique.",
        ),
        ModelProfile(
            name="grok",
            model_id="grok-1",
            context_window=8192,
            params=GenerationParams(temperature=0.5, max_tokens=1024),
            notes="xAI Grok-1 (open weights, 314B MoE) — large; enable if you can host it.",
        ),
    ]


def build_default_registry(policy: AirgapPolicy | None = None) -> ModelRegistry:
    reg = ModelRegistry(policy=policy)
    for profile in default_profiles():
        reg.register(profile)
    return reg
