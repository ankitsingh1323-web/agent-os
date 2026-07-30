# Agent OS

**An airgapped operating system for coordinating AI agents on local open-weight models.**

Agent OS treats a fleet of general-purpose agents like processes in an operating
system: a **kernel** schedules them, **routes** work to them, brokers every
**syscall** (inference, tools, memory, messaging, spawning), and enforces a hard
**airgap** — all model inference runs against *local* open-weight runtimes
(**Kimi, Qwen, Llama, Grok**). No cloud APIs. No egress. No telemetry.

The core is **standard-library only**, so it installs nothing and runs inside a
real airgap. With no model server running it degrades to a deterministic offline
stub, so you can see the whole system work before loading a single weight.

---

## Quickstart

```bash
# No install needed — core is stdlib-only.
python -m agentos info
python -m agentos run "Design a nightly backup strategy for an airgapped file server"

# Or the scripted demo (runs fully offline via the stub backend):
python examples/demo.py

# Tests (dev extra):
pip install -e ".[dev]" && pytest
```

Programmatic use:

```python
import asyncio
from agentos import build_default_os

os_ = build_default_os(state_dir="./.agentos_state")
report = asyncio.run(os_.run("Plan and review a small feature"))
print(report.answer)
print(report.trace())   # full plan → execute → review → summarize trace
```

## Running on real local models

Serve the open-weight models with any OpenAI-compatible local runtime and the OS
will use them automatically (the `auto` backend probes `localhost` and falls back
to the stub if nothing answers):

```bash
# Ollama example
ollama serve
ollama pull qwen2.5:14b && ollama pull llama3.1:8b   # + kimi / grok as available
```

Then point/re-bind models via `config/agentos.example.json`:

```bash
python -m agentos --config config/agentos.example.json run "…"
```

Qwen/Llama run comfortably on a workstation. Kimi-K2 and Grok-1 (314B MoE) are
large — enable them where you have the hardware; everything else keeps working
regardless.

## The agent fleet

| Agent | Model | Role |
|-----------|--------|-----------------------------------------------|
| planner | kimi | Decompose a goal into a dependency-ordered plan |
| researcher| qwen | Gather facts, constraints, prior art (local only) |
| coder | qwen | Produce a concrete solution / draft |
| reviewer | llama | Check correctness, risk, completeness |
| critic | llama | Adversarially probe for failure modes |
| summarizer| kimi | Synthesize everything into the final answer |

Re-bind any agent to any model in config — no code changes.

## Airgap guarantee

Every backend calls `assert_local()` before opening a socket. Non-loopback /
non-private hosts are refused with an `AirgapViolation`; proxies are bypassed.
The policy is strict by default and only relaxes if you explicitly say so.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full design.

## License

Apache-2.0.
