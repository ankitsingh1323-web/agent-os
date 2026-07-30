"""Core tests for the airgapped Agent OS. Run with: python -m pytest -q
(also runnable directly: python tests/test_agentos.py)."""

import asyncio

import pytest

from agentos import build_default_os
from agentos.inference.airgap import AirgapPolicy, AirgapViolation, assert_local
from agentos.inference.backend import GenerationParams, StubBackend
from agentos.kernel.task import TaskStatus
from agentos.memory.store import MemoryRecord, MemoryStore
from agentos.tools.registry import build_default_tools


# --- airgap -----------------------------------------------------------------
def test_airgap_allows_localhost():
    assert_local("http://localhost:11434/v1")
    assert_local("http://127.0.0.1:8000")
    assert_local("http://192.168.1.10:11434")  # private LAN


def test_airgap_blocks_public():
    with pytest.raises(AirgapViolation):
        assert_local("https://api.openai.com/v1")
    with pytest.raises(AirgapViolation):
        assert_local("http://8.8.8.8/x")


def test_airgap_can_be_disabled():
    assert_local("https://example.com", AirgapPolicy(enforce=False))


# --- stub backend -----------------------------------------------------------
def test_stub_is_role_aware():
    out = asyncio.run(
        StubBackend().generate(
            [{"role": "system", "content": "[ROLE: planner]"},
             {"role": "user", "content": "do a thing"}],
            GenerationParams(),
            "m",
        )
    )
    assert '"steps"' in out  # planner emits a JSON plan


# --- tools ------------------------------------------------------------------
def test_tool_calculator():
    reg = build_default_tools()
    assert reg.get("calculator").run(expression="2 + 3 * 4").output == 14


def test_tool_sandbox_blocks_escape(tmp_path):
    reg = build_default_tools(workspace=str(tmp_path))
    res = reg.get("read_file").run(path="../../etc/passwd")
    assert res.ok is False


# --- memory -----------------------------------------------------------------
def test_memory_keyword_search():
    store = MemoryStore()
    store.add(MemoryRecord(id="1", kind="fact", text="airgapped backups need rotation"))
    store.add(MemoryRecord(id="2", kind="fact", text="unrelated note about cats"))
    hits = store.search("backup rotation", k=5)
    assert hits and hits[0].id == "1"


# --- end-to-end orchestration ----------------------------------------------
def test_end_to_end_run():
    os_ = build_default_os()
    report = asyncio.run(os_.run("Plan and review a small feature."))
    statuses = {t.status for t in report.tasks}
    assert TaskStatus.DONE in statuses
    assert report.answer  # summarizer produced something
    assert report.plan     # planner produced steps


def test_permission_ceiling_blocks_unlisted_tool():
    os_ = build_default_os()
    reviewer = os_.kernel.agents.get("reviewer")  # has no allowed_tools
    res = os_.kernel.run_tool(reviewer, "read_file", path="x")
    assert res.ok is False


# --- external agent adapter -------------------------------------------------
def test_external_agent_runs_under_kernel():
    from agentos.agents import adapt

    async def runner(harness):
        out = await harness.infer(f"handle: {harness.goal}")
        harness.remember(out)
        return out

    os_ = build_default_os()
    os_.kernel.agents.register(adapt("ext", runner, model="qwen", role="researcher"))
    task = os_.kernel.new_task("do the thing", agent="ext")
    asyncio.run(os_.kernel.run_task(task))
    assert task.status is TaskStatus.DONE
    assert task.result
    assert any(e.kind == "infer" for e in os_.kernel.audit.events())  # routed through kernel


def test_egress_guard_blocks_rogue_external_agent():
    import socket
    from agentos.agents import adapt

    def rogue(harness):
        socket.create_connection(("1.1.1.1", 53), timeout=1)  # own I/O — forbidden
        return "leaked"

    os_ = build_default_os()
    os_.kernel.agents.register(adapt("rogue", rogue, role="external"))
    task = os_.kernel.new_task("phone home", agent="rogue")
    asyncio.run(os_.kernel.run_task(task))
    assert task.status is TaskStatus.FAILED
    assert "Airgap" in (task.error or "")


def test_egress_guard_allows_local():
    import socket
    from agentos.inference.airgap import AirgapViolation, process_egress_guard

    with process_egress_guard():
        try:
            socket.create_connection(("127.0.0.1", 9), timeout=0.2)  # likely refused
        except AirgapViolation:
            raise AssertionError("local connection must not be blocked")
        except OSError:
            pass  # connection refused/timeout is fine — the point is it wasn't airgap-blocked


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
