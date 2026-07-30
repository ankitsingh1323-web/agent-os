"""Tool registry and built-in local tools.

Every tool is local by construction — filesystem, arithmetic, document search.
Nothing here reaches the network, which keeps the airgap intact even as agents
gain capabilities. Tools declare a permission tier; the kernel decides what an
agent may call.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List


@dataclass
class ToolResult:
    ok: bool
    output: object
    error: str = ""


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., ToolResult]
    permission: str = "safe"  # "safe" | "workspace" | "privileged"
    schema: dict = field(default_factory=dict)

    def run(self, **kwargs) -> ToolResult:
        try:
            return self.func(**kwargs)
        except Exception as exc:  # tools must never crash the kernel
            return ToolResult(ok=False, output=None, error=f"{type(exc).__name__}: {exc}")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'. Known: {sorted(self._tools)}")
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools)

    def describe(self) -> List[dict]:
        return [
            {"name": t.name, "description": t.description, "permission": t.permission}
            for t in self._tools.values()
        ]


# --- built-in tools ---------------------------------------------------------

_ARITH = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH:
        return _ARITH[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH:
        return _ARITH[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def _calculator(expression: str) -> ToolResult:
    value = _safe_eval(ast.parse(expression, mode="eval"))
    return ToolResult(ok=True, output=value)


def build_default_tools(workspace: str = ".") -> ToolRegistry:
    """Register the safe local toolset, sandboxed to ``workspace``."""
    root = Path(workspace).resolve()
    reg = ToolRegistry()

    def _resolve(path: str) -> Path:
        target = (root / path).resolve()
        if root not in target.parents and target != root:
            raise PermissionError(f"path escapes workspace: {path}")
        return target

    def read_file(path: str) -> ToolResult:
        return ToolResult(ok=True, output=_resolve(path).read_text())

    def write_file(path: str, content: str) -> ToolResult:
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return ToolResult(ok=True, output=f"wrote {len(content)} bytes to {path}")

    def list_dir(path: str = ".") -> ToolResult:
        return ToolResult(ok=True, output=sorted(p.name for p in _resolve(path).iterdir()))

    reg.register(Tool("calculator", "Evaluate a basic arithmetic expression.", _calculator))
    reg.register(Tool("read_file", "Read a UTF-8 file inside the workspace.", read_file, "workspace"))
    reg.register(Tool("write_file", "Write a UTF-8 file inside the workspace.", write_file, "workspace"))
    reg.register(Tool("list_dir", "List a directory inside the workspace.", list_dir, "workspace"))
    return reg
