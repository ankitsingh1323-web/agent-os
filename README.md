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

## Run on a real model right now (no downloads)

The OS ships a real, self-contained model — **NanoLM**, a trained-at-startup
n-gram LM with genuine temperature/top-p sampling, pure stdlib, CPU-only. It runs
inside a true airgap with nothing to install, and it serves the **same
OpenAI-compatible API** as the production runtimes.

```bash
# in-process real model (no server): pick the "nano" backend
python -c "import asyncio; from agentos import build_default_os; from agentos.inference.registry import ModelProfile as P; \
os=build_default_os(); [os.kernel.models.register(P(name=n, model_id='nano', backend='nano')) for n in os.kernel.models.names()]; \
print(asyncio.run(os.run('Design a nightly backup plan')).answer)"

# or serve it over HTTP and let the OS discover it, exactly like Ollama:
python -m agentos.inference.server        # NanoLM on 127.0.0.1:11434
python -m agentos run "Design a nightly backup plan"   # auto-detects the server

# full socket → HTTP → kernel → model demo:
python examples/local_model_demo.py
```

NanoLM is a compact *reference* model that proves the OS runs on locally computed
inference — not a replacement for the big open-weight models. Those plug into the
identical path below.

## Running on real production models

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

## Integrating externally-built agents

The OS uses a **narrow-waist** contract: a thin, mandatory interface at the
boundary, total freedom behind it. An agent built any other way — LangChain,
AutoGen, CrewAI, a raw prompt loop — runs under the kernel by writing a small
runner against the **harness** (the framework-agnostic "syscall SDK"). Its model
calls, tools and memory all flow through the kernel, so the airgap, audit trail
and permissions still hold.

```python
from agentos import build_default_os
from agentos.agents import adapt

async def my_runner(harness):            # the entire integration surface
    findings = await harness.infer(f"Research: {harness.goal}")   # sanctioned model path
    harness.remember(findings)
    return findings

os_ = build_default_os()
os_.kernel.agents.register(adapt("field-researcher", my_runner, model="qwen",
                                 role="researcher", capabilities=["research"]))
```

The one iron rule — **no agent performs its own I/O** — is *enforced*, not
trusted: foreign runners execute inside a process egress guard, so any attempt at
non-local network access is refused at the socket. See
`examples/external_agent_demo.py` (a foreign agent runs; a rogue one is blocked).

## Airgap guarantee

Every backend calls `assert_local()` before opening a socket. Non-loopback /
non-private hosts are refused with an `AirgapViolation`; proxies are bypassed.
The policy is strict by default and only relaxes if you explicitly say so.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full design.

## License

Apache-2.0.
