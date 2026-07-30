"""Optional JSON configuration.

Everything has a working code default, so config is purely for overriding: point
models at real local runtimes, re-bind agents to different models, widen/narrow
the airgap policy, or change concurrency — all without touching code. JSON is used
(not YAML) so the core stays stdlib-only and installs nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from agentos.inference.airgap import AirgapPolicy
from agentos.inference.backend import GenerationParams
from agentos.inference.registry import ModelProfile


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def policy_from(cfg: dict) -> Optional[AirgapPolicy]:
    ag = cfg.get("airgap")
    if ag is None:
        return None
    return AirgapPolicy(
        enforce=ag.get("enforce", True),
        allow_private_lan=ag.get("allow_private_lan", True),
        extra_hosts=set(ag.get("extra_hosts", [])),
    )


def profiles_from(cfg: dict) -> List[ModelProfile]:
    out = []
    for m in cfg.get("models", []):
        params = m.get("params", {})
        out.append(
            ModelProfile(
                name=m["name"],
                model_id=m["model_id"],
                backend=m.get("backend", "auto"),
                base_url=m.get("base_url", "http://localhost:11434/v1"),
                context_window=m.get("context_window", 8192),
                params=GenerationParams(
                    temperature=params.get("temperature", 0.7),
                    max_tokens=params.get("max_tokens", 1024),
                    top_p=params.get("top_p", 0.95),
                    stop=params.get("stop", []),
                ),
                notes=m.get("notes", ""),
            )
        )
    return out


def agent_models_from(cfg: dict) -> Dict[str, str]:
    return dict(cfg.get("agent_models", {}))
