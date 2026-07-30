"""Agent OS — an airgapped operating system for coordinating AI agents.

Everything runs against locally-hosted open-weight models (Kimi, Qwen, Llama,
Grok). No cloud APIs, no telemetry, no egress. The core depends only on the
Python standard library so it runs inside a real airgap with zero installs.
"""

__version__ = "0.1.0"

from agentos.bootstrap import build_default_os  # noqa: E402

__all__ = ["build_default_os", "__version__"]
