"""Inference layer: local, airgapped access to open-weight model runtimes."""

from agentos.inference.airgap import AirgapPolicy, AirgapViolation, assert_local
from agentos.inference.backend import (
    AutoBackend,
    Backend,
    GenerationParams,
    LocalOpenAIBackend,
    StubBackend,
)
from agentos.inference.registry import ModelProfile, ModelRegistry

__all__ = [
    "AirgapPolicy",
    "AirgapViolation",
    "assert_local",
    "Backend",
    "AutoBackend",
    "LocalOpenAIBackend",
    "StubBackend",
    "GenerationParams",
    "ModelProfile",
    "ModelRegistry",
]
