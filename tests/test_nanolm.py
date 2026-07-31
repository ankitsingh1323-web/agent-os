"""Tests for the working local model: NanoLM, its server, and the OS on top."""

import asyncio
import json
import urllib.request

from agentos import build_default_os
from agentos.inference.backend import GenerationParams
from agentos.inference.nanolm import NanoLM, NanoLMBackend
from agentos.inference.registry import ModelProfile
from agentos.inference.server import serve_in_thread


def test_nanolm_generates_real_text():
    m = NanoLM()
    out = m.complete("[ROLE: researcher] you are the researcher.",
                     "backup strategy for an airgapped server",
                     GenerationParams(temperature=0.7, max_tokens=40), seed=1)
    assert isinstance(out, str) and len(out.split()) >= 4
    assert out[0].isupper() and out.rstrip()[-1] in ".!?"   # well-formed


def test_nanolm_is_deterministic_with_seed():
    m = NanoLM()
    a = m.complete("[ROLE: coder]", "implement snapshot rotation",
                   GenerationParams(max_tokens=30), seed=42)
    b = m.complete("[ROLE: coder]", "implement snapshot rotation",
                   GenerationParams(max_tokens=30), seed=42)
    assert a == b


def test_nanolm_backend_via_kernel():
    m = NanoLM()
    out = asyncio.run(NanoLMBackend(m).generate(
        [{"role": "system", "content": "[ROLE: reviewer]"},
         {"role": "user", "content": "review the plan"}],
        GenerationParams(), "nano"))
    assert isinstance(out, str) and out


def test_server_speaks_openai_and_stays_local():
    server, base_url = serve_in_thread(port=0)
    try:
        body = json.dumps({
            "model": "nano",
            "messages": [
                {"role": "system", "content": "[ROLE: researcher]"},
                {"role": "user", "content": "airgapped backups"},
            ],
        }).encode()
        req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        payload = json.loads(urllib.request.urlopen(req, timeout=5).read())
        assert payload["object"] == "chat.completion"
        assert payload["choices"][0]["message"]["content"]
    finally:
        server.shutdown()


def test_os_runs_on_live_local_server():
    server, base_url = serve_in_thread(port=0)
    try:
        os_ = build_default_os()
        for name in os_.kernel.models.names():
            os_.kernel.models.register(
                ModelProfile(name=name, model_id="nano", backend="local-openai", base_url=base_url))
        task = os_.kernel.new_task("Research replication options.", agent="researcher")
        asyncio.run(os_.kernel.run_task(task))
        assert task.result and task.status.value == "done"
    finally:
        server.shutdown()
