"""NanoLM — a real, self-contained language model that runs airgapped.

The stub backend returns canned strings; NanoLM actually *generates*. It is a
variable-order word n-gram model with stupid-backoff and true temperature / top-p
(nucleus) sampling, trained at import time on a small bundled corpus. Pure stdlib,
CPU-only, no downloads — so it works inside a real airgap and in CI.

It is deliberately small: a *reference* model that proves the OS runs on locally
computed inference, not a replacement for Kimi/Qwen/Llama/Grok. Those production
models plug in over the identical OpenAI path (see ``server.py`` and
``LocalOpenAIBackend``) — swapping them in is a config change, not a code change.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from agentos.inference.backend import Backend, GenerationParams, Message

_SENT_END = {".", "!", "?"}
_CLOSE = {".", ",", ";", ":", "!", "?", ")"}
_STOP = {
    "the", "a", "an", "of", "to", "and", "for", "in", "on", "is", "are", "be",
    "it", "that", "this", "with", "as", "at", "by", "or", "from", "into", "you",
}
# Names to re-capitalize on output.
_PROPER = {"kimi", "qwen", "llama", "grok", "ollama", "vllm", "agent", "os"}
# Nudge each agent role toward on-topic vocabulary when the prompt is thin.
_ROLE_SEED = {
    "planner": "plan", "researcher": "research", "coder": "solution",
    "reviewer": "review", "critic": "risk", "summarizer": "summary",
}

# A compact domain corpus. Richer text → more coherent generation.
DEFAULT_CORPUS = """
An agent operating system coordinates many agents like processes. The kernel
schedules each task, routes it to the right agent, and brokers every request
through a small set of syscalls. Agents never touch the network directly. Every
model call passes through the airgap guard before a socket opens, so nothing
leaves the machine. The airgap is an invariant, not an option.

A backup strategy for an airgapped file server should run nightly. Snapshots are
rotated and verified. Keep at least three copies on separate media. Test the
restore path regularly, because an untested backup is only a hope. Encrypt the
archive at rest and record every operation in the local audit log.

The planner decomposes a goal into a plan of small tasks with clear dependencies.
The researcher gathers the relevant facts and constraints. The coder produces a
concrete solution and guards the edges. The reviewer checks the draft for
correctness and risk, and returns a verdict. The critic probes for failure modes
and blind spots. The summarizer integrates the results into one clear answer.

Local open weight models run on the machine itself. Kimi handles long context
reasoning and planning. Qwen is strong for research and coding. Llama gives
balanced judgement for review. Grok is large and needs a capable host. Each model
is served locally and bound to an agent by name, so any agent can be repointed.

Security depends on least privilege. Tools are sandboxed to a workspace. Every
syscall is recorded so the run can be reviewed later. Permission is granted
narrowly and revoked quickly. A privileged action pauses for human approval. Risk
is named plainly and mitigated with a concrete step.

Memory has three tiers. Working memory is the scratchpad for one task. The shared
blackboard lets agents post partial results for others to build on. The long term
store keeps facts and decisions that survive a restart. Retrieval brings back what
is relevant to the goal at hand.
"""


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+|[.,;:!?()]", text.lower())


class NanoLM:
    """A word-level n-gram language model with backoff and nucleus sampling."""

    def __init__(self, corpus: str = DEFAULT_CORPUS, order: int = 3) -> None:
        self.order = max(2, order)
        toks = _tokenize(corpus)
        self.tokens = toks
        self.vocab = set(toks)
        # ngrams[k]: context of length k -> Counter of following tokens
        self.ngrams: Dict[int, Dict[Tuple[str, ...], Counter]] = {
            k: defaultdict(Counter) for k in range(1, self.order)
        }
        self.unigram: Counter = Counter(toks)
        for i in range(len(toks)):
            for k in range(1, self.order):
                if i - k >= 0:
                    ctx = tuple(toks[i - k:i])
                    self.ngrams[k][ctx][toks[i]] += 1
        # sentence-start contexts (tokens following a sentence end)
        self.starts: List[Tuple[str, ...]] = []
        for i in range(1, len(toks) - 1):
            if toks[i - 1] in _SENT_END:
                self.starts.append(tuple(toks[i:i + self.order - 1]))
        if not self.starts:
            self.starts = [tuple(toks[:self.order - 1])]

    # --- sampling ----------------------------------------------------------
    def _sample(self, counter: Counter, temperature: float, top_p: float, rng: random.Random) -> str:
        total = sum(counter.values())
        dist = [(tok, c / total) for tok, c in counter.items()]
        if temperature and temperature > 0:
            inv = 1.0 / temperature
            dist = [(tok, p ** inv) for tok, p in dist]
        s = sum(p for _, p in dist) or 1.0
        dist = [(tok, p / s) for tok, p in dist]
        dist.sort(key=lambda x: x[1], reverse=True)
        # nucleus (top-p)
        nucleus, cum = [], 0.0
        for tok, p in dist:
            nucleus.append((tok, p))
            cum += p
            if cum >= top_p:
                break
        s = sum(p for _, p in nucleus) or 1.0
        r = rng.random() * s
        acc = 0.0
        for tok, p in nucleus:
            acc += p
            if r <= acc:
                return tok
        return nucleus[-1][0]

    def _next_counter(self, ctx: List[str]) -> Counter:
        for k in range(min(len(ctx), self.order - 1), 0, -1):
            c = self.ngrams[k].get(tuple(ctx[-k:]))
            if c:
                return c
        return self.unigram

    def generate(
        self,
        seed_ctx: List[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        rng: random.Random,
        max_sentences: int = 3,
    ) -> List[str]:
        budget = max(8, min(max_tokens, 80))
        ctx = list(seed_ctx)
        out: List[str] = []
        sentences = 0
        for _ in range(budget):
            nxt = self._sample(self._next_counter(ctx), temperature, top_p, rng)
            ctx.append(nxt)
            out.append(nxt)
            if nxt in _SENT_END:
                sentences += 1
                if sentences >= max_sentences:
                    break
        return out

    # --- prompt-conditioned completion ------------------------------------
    def complete(
        self,
        system: str,
        user: str,
        params: GenerationParams,
        seed: Optional[int] = None,
    ) -> str:
        rng = random.Random(seed if seed is not None else (hash((system, user)) & 0xFFFFFFFF))
        role = ""
        if "[ROLE:" in system:
            role = system.split("[ROLE:", 1)[1].split("]", 1)[0].strip().lower()

        # Seed context: salient prompt words present in the vocab, plus a role nudge.
        salient = [t for t in _tokenize(user) if t in self.vocab and t not in _STOP and len(t) > 3]
        if role in _ROLE_SEED and _ROLE_SEED[role] in self.vocab:
            salient.append(_ROLE_SEED[role])
        if salient:
            seed_ctx = [rng.choice(salient[-5:])]
        else:
            seed_ctx = list(rng.choice(self.starts))

        toks = self.generate(
            seed_ctx,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            rng=rng,
        )
        return _detokenize(toks)


def _detokenize(toks: List[str]) -> str:
    out = ""
    for i, t in enumerate(toks):
        if t in _CLOSE:
            out += t
        elif t == "(":
            out += (" " if out and not out.endswith("(") else "") + t
        else:
            out += ("" if (not out or out.endswith("(")) else " ") + t
    out = out.strip()
    # Capitalize sentence starts.
    out = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    # Re-capitalize known proper nouns.
    for name in _PROPER:
        out = re.sub(rf"\b{name}\b", name.capitalize(), out)
    if out and out[-1] not in _SENT_END:
        out += "."
    return out or "…"


class NanoLMBackend(Backend):
    """In-process backend wrapping a shared NanoLM. The simplest 'working model':
    real generation with no server and no network at all."""

    name = "nano"
    _shared: Optional[NanoLM] = None

    def __init__(self, model: Optional[NanoLM] = None) -> None:
        if model is not None:
            self._model = model
        else:
            if NanoLMBackend._shared is None:
                NanoLMBackend._shared = NanoLM()
            self._model = NanoLMBackend._shared

    async def generate(self, messages: List[Message], params: GenerationParams, model_id: str) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return self._model.complete(system, user, params)
