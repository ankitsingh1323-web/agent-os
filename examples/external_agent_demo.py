"""Integrating externally-built agents under the kernel.

Runs fully offline. Demonstrates the narrow-waist contract:

  1. A foreign "framework" agent (imagine a LangChain/AutoGen chain) is adapted
     with a tiny runner and scheduled by the kernel like any built-in agent —
     its model call is transparently routed through the airgap + audit.
  2. A *rogue* external agent that tries its own outbound network call is refused
     at the socket by the process egress guard. No self-I/O — enforced, not asked.

    python examples/external_agent_demo.py
"""

import asyncio
import socket

from agentos import build_default_os
from agentos.agents import adapt
from agentos.inference.airgap import AirgapViolation


# --- 1. a foreign agent, as someone else might have built it ---------------
class PretendLangChainAgent:
    """Stand-in for a framework agent. Normally it would call its *own* LLM
    client; to run under the OS we simply point it at the harness instead."""

    def __init__(self, style: str) -> None:
        self.style = style

    async def run(self, harness) -> str:
        prompt = f"({self.style} style) Investigate and report on: {harness.goal}"
        answer = await harness.infer(prompt)          # sanctioned model path
        harness.remember(answer, kind="external-research")
        return answer


def make_research_runner():
    chain = PretendLangChainAgent(style="concise, evidence-first")

    async def runner(harness):                        # the whole integration surface
        return await chain.run(harness)

    return runner


# --- 2. a rogue agent that tries to phone home -----------------------------
def rogue_runner(harness):
    # Direct IP (no DNS) so we exercise OUR guard, not name resolution.
    socket.create_connection(("1.1.1.1", 53), timeout=1)
    return "exfiltrated the data"


async def main() -> None:
    os_ = build_default_os(state_dir="./.agentos_state")

    # Register both external agents alongside the built-in fleet.
    os_.kernel.agents.register(
        adapt(
            "field-researcher",
            make_research_runner(),
            model="qwen",
            role="researcher",
            description="Externally-built research agent (adapted).",
            capabilities=["research"],
        )
    )
    os_.kernel.agents.register(
        adapt("rogue", rogue_runner, model="llama", role="external")
    )

    print("=" * 68)
    print("EXTERNAL AGENTS UNDER THE KERNEL")
    print("=" * 68)

    # 1. Foreign agent runs like a native one.
    task = os_.kernel.new_task(
        "Assess offsite replication options for an airgapped archive.",
        agent="field-researcher",
    )
    await os_.kernel.run_task(task)
    print(f"\n[{task.status.value.upper()}] field-researcher (adapted foreign agent):")
    print("    " + (task.result or task.error or "").replace("\n", "\n    "))

    # 2. Rogue agent is stopped at the socket.
    rogue = os_.kernel.new_task("Send the archive to an external host.", agent="rogue")
    await os_.kernel.run_task(rogue)
    print(f"\n[{rogue.status.value.upper()}] rogue (attempted self-I/O):")
    print(f"    airgap held → {rogue.error}")

    # The guard is a hard guarantee, shown directly:
    try:
        from agentos.inference.airgap import process_egress_guard
        with process_egress_guard():
            socket.create_connection(("8.8.8.8", 53), timeout=1)
        print("\n(!) egress guard did NOT block — unexpected")
    except AirgapViolation as exc:
        print(f"\nDirect proof: {exc}")

    print(f"\nAudit events: {len(os_.kernel.audit.events())} "
          f"(every syscall from the foreign agent was recorded)")


if __name__ == "__main__":
    asyncio.run(main())
