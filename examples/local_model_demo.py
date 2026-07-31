"""Run the OS on a real, locally-served model — airgapped, no downloads.

This exercises the *production inference path*, not the stub:

  1. Boots the NanoLM OpenAI-compatible server on 127.0.0.1 (a free port).
  2. Points every agent's model at it via LocalOpenAIBackend.
  3. Sends a real HTTP completion over the socket (airgap guard permits loopback).
  4. Runs a real agent task end-to-end through the kernel on that model.

Swap NanoLM's server for Ollama/vLLM serving Qwen or Llama and nothing here
changes — that is the point of the OpenAI-compatible seam.

    python examples/local_model_demo.py
"""

import asyncio

from agentos import build_default_os
from agentos.inference.registry import ModelProfile
from agentos.inference.server import serve_in_thread


async def main() -> None:
    # 1. Stand up the real local model server.
    server, base_url = serve_in_thread(port=0)
    print("=" * 68)
    print(f"Local model server (NanoLM, OpenAI API) → {base_url}")
    print("=" * 68)

    try:
        os_ = build_default_os()

        # 2. Repoint every model at the live server via the real HTTP backend.
        for name in os_.kernel.models.names():
            os_.kernel.models.register(
                ModelProfile(name=name, model_id="nano", backend="local-openai", base_url=base_url)
            )

        # 3. Raw completion — proves the socket + HTTP + airgap-guarded path works.
        text = await os_.kernel.infer(
            "qwen",
            [
                {"role": "system", "content": "[ROLE: researcher] you are the researcher."},
                {"role": "user", "content": "backup strategy for an airgapped file server"},
            ],
            None,
        )
        print("\nDirect model call over HTTP (127.0.0.1):")
        print("    " + text)

        # 4. A real agent task, routed through the kernel onto the live model.
        task = os_.kernel.new_task(
            "Research offsite options for an airgapped archive.", agent="researcher"
        )
        await os_.kernel.run_task(task)
        print(f"\n[{task.status.value.upper()}] researcher (on the live local model):")
        print("    " + (task.result or task.error or ""))

        print(f"\nAll inference stayed on {base_url} — no external egress, "
              f"{len(os_.kernel.audit.events())} syscalls audited.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
