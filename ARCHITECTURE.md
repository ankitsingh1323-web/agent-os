# Agent OS — Architecture

> An **airgapped** operating system for coordinating AI agents on **local
> open-weight models** (Kimi, Qwen, Llama, Grok).

This document is the architectural decision record. It states the goals, the
constraints, the layered design, the key abstractions, and the trade-offs.

---

## 1. Design goals

1. **Airgapped by construction.** No inference, retrieval, or telemetry ever
   leaves the machine. This is an invariant, not a config toggle you remember to
   set — the default is strict and enforcement lives at the socket boundary.
2. **Open-weight, local models.** Kimi, Qwen, Llama, Grok — served by any local
   OpenAI-compatible runtime (Ollama, vLLM, llama.cpp, LM Studio). Models are
   swappable profiles, bound to agents by name.
3. **General-purpose agent fleet.** A small set of role agents (planner,
   researcher, coder, reviewer, critic, summarizer) that compose to handle
   open-ended goals — the "agent list" that any general task can draw on.
4. **OS-like structure.** A privileged kernel; agents in "user space" that reach
   resources only through brokered syscalls; a scheduler, router, message bus,
   memory hierarchy, and an audit trail.
5. **Zero-dependency core.** Standard library only, so it installs and runs in a
   true airgap. It stays fully demonstrable with **no weights loaded** via a
   deterministic stub backend.

## 2. Constraints that shaped the design

| Constraint | Consequence |
|---|---|
| No network egress | Airgap guard at every socket; no cloud SDKs; JSON not YAML (no PyYAML); keyword memory, not a hosted vector DB. |
| Local models are the substrate | Provider-agnostic OpenAI-compatible client; per-agent model binding; `auto` backend that survives a missing server. |
| Must run without weights | `StubBackend` gives deterministic, role-aware output so the OS + tests run offline. |
| Accountability offline | Local append-only audit log of every syscall. |

## 3. Layered architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION (policy)   Orchestrator: plan → schedule → execute →    │
│                           summarize   (swappable: debate, pipeline…)   │
├──────────────────────────────────────────────────────────────────────┤
│  AGENTS (user space)      planner · researcher · coder · reviewer ·    │
│                           critic · summarizer     (Agent + Registry)   │
│        │ syscalls only (infer / tool / memory / emit / spawn)          │
├────────┼─────────────────────────────────────────────────────────────┤
│  KERNEL (privileged)   AgentContext ── Router ── Scheduler ── Perms ── │
│                        Audit    (owns every subsystem below)           │
├──────────────────────────────────────────────────────────────────────┤
│ Inference   │ Memory        │ Bus        │ Tools      │ Telemetry      │
│ ModelReg    │ Working /     │ pub/sub    │ registry + │ audit log      │
│ Backends    │ Blackboard /  │ topics     │ sandboxed  │ (local, append │
│ Airgap grd  │ Store (KW)    │            │ local ops  │  only)         │
├──────────────────────────────────────────────────────────────────────┤
│  MODEL RUNTIME (local, airgapped)   Ollama · vLLM · llama.cpp · LMStudio│
│  kimi (Kimi-K2) · qwen (Qwen2.5) · llama (Llama-3.1) · grok (Grok-1)   │
└──────────────────────────────────────────────────────────────────────┘
                     ▲ assert_local() gates every socket ▲
```

## 4. Components

### 4.1 Inference layer — `agentos/inference/`
- **`airgap.py`** — `AirgapPolicy` + `assert_local()`. The choke point: loopback
  and RFC-1918 private hosts allowed; everything else raises `AirgapViolation`.
  DNS is treated as egress (unknown hostnames are refused). Proxies are bypassed.
- **`backend.py`** — `Backend` interface and three implementations:
  - `LocalOpenAIBackend` — `/chat/completions` over stdlib `urllib`, pinned local.
  - `StubBackend` — deterministic, role-aware; keys off a `[ROLE: …]` marker so
    the offline demo and tests produce coherent output with no weights.
  - `AutoBackend` — probes the local endpoint once; uses it if up, else the stub.
- **`registry.py`** — `ModelProfile` + `ModelRegistry`. Logical names
  (`kimi/qwen/llama/grok`) → runtime, model id, sampling params. Lazy backend
  construction.

### 4.2 Kernel — `agentos/kernel/`
- **`kernel.py`** — owns every subsystem; serves syscalls (`infer`, `run_tool`,
  `spawn`); enforces the tool **permission ceiling**; records the audit trail;
  runs the task lifecycle.
- **`context.py`** — `AgentContext`, the syscall surface. Agents touch nothing
  else. This boundary is what makes a pile of agents an *operating system*.
- **`scheduler.py`** — dependency-aware wave scheduler with bounded concurrency;
  threads each finished task's result into its dependents; fails closed on an
  unsatisfiable graph.
- **`router.py`** — explicit assignment wins; otherwise capability/keyword match.
- **`orchestrator.py`** — the default top-level policy (plan → execute → summarize)
  and a `RunReport` with a human-readable trace.

### 4.3 Agents — `agentos/agents/`
`Agent` = role + bound model + capability/tool set, acting only via syscalls.
Dependency results are threaded into each agent's prompt automatically. The
built-in fleet is general-purpose and composable; add your own by subclassing
`Agent` and registering it.

### 4.4 Memory — `agentos/memory/`
Three tiers: **WorkingMemory** (per-task scratch), **Blackboard** (shared,
persisted JSON), **MemoryStore** (long-term, append-only JSONL, keyword
retrieval). The store interface is intentionally small so a local vector backend
can replace the keyword retriever without touching agents.

### 4.5 Bus, Tools, Telemetry
- **Bus** — in-process async pub/sub; one auditable place for inter-agent messages.
- **Tools** — registry of **local** capabilities (calculator, workspace-sandboxed
  file ops). Each declares a permission tier; the kernel enforces a ceiling.
- **Telemetry** — append-only local audit log of every syscall.

## 5. Execution model

1. A goal enters the **Orchestrator**.
2. The **planner** (kimi) emits a JSON plan of sub-tasks with dependencies.
3. Steps become **Tasks**; the **Scheduler** runs ready tasks concurrently,
   threading upstream results forward.
4. The **Router** binds each task to an agent; the agent runs via **syscalls**.
5. The **summarizer** (kimi) synthesizes all results into the final answer.
6. Every step is **audited** locally.

## 6. Key trade-offs

- **Stdlib-only vs. features.** We forgo hosted vector search and YAML to
  guarantee airgap install. Extension points (memory store, backend, orchestrator,
  router) let you add heavier machinery deliberately.
- **Stub backend.** Makes the system runnable/testable offline; clearly marked so
  stub output is never mistaken for a model's.
- **Wave scheduler vs. streaming DAG.** Simple, predictable, easy to audit; a
  fully streaming executor is a drop-in future upgrade behind the same interface.
- **Deterministic router first.** Fast and offline for the common case; an
  LLM-router is an alternative strategy behind `Router`.

## 7. Extension points

| Want to… | Implement / override |
|---|---|
| Add a model runtime | `Backend` subclass + `ModelProfile` |
| Add an agent | subclass `Agent`, register it |
| Add a capability | register a `Tool` (keep it local) |
| Change coordination | new orchestrator over the same kernel syscalls |
| Real semantic memory | swap `MemoryStore` for a local-vector implementation |
| Relax/adjust the airgap | `AirgapPolicy` (explicit + audited) |

## 8. Security & governance

- Airgap enforced at the socket boundary; proxies bypassed; DNS treated as egress.
- Tools are sandboxed to a workspace root; path escapes are refused.
- Per-agent tool allow-lists + a kernel permission ceiling.
- Full local audit trail for after-the-fact review.
