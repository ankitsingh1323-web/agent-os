"""End-to-end demo of the airgapped Agent OS.

Runs entirely offline: with no local model server up, the OS transparently uses
the deterministic stub backend, so this prints a full plan → execute → review →
summarize trace with zero setup.

    python examples/demo.py
"""

import asyncio

from agentos import build_default_os


async def main() -> None:
    os_ = build_default_os(state_dir="./.agentos_state")

    print("=" * 70)
    print("AGENT OS — airgapped, local open-weight models")
    print("=" * 70)
    for agent in os_.describe()["agents"]:
        print(f"  · {agent['name']:12} model={agent['model']:6} — {agent['description']}")
    print()

    goal = "Design a nightly backup strategy for an airgapped file server."
    report = await os_.run(goal)
    print(report.trace())

    print("\n" + "-" * 70)
    print(f"Audit events recorded: {len(os_.kernel.audit.events())}")
    print(f"Bus messages: {len(os_.kernel.bus.history())}")
    print(f"Long-term memories: {len(os_.kernel.memory.all())}")


if __name__ == "__main__":
    asyncio.run(main())
