"""A local, OpenAI-compatible inference server backed by NanoLM.

This exposes ``POST /v1/chat/completions`` — the exact API Ollama, vLLM and
llama.cpp-server speak — so the OS's ``LocalOpenAIBackend`` talks to it over a
real socket with the real HTTP protocol. It is the proof that the production
inference path works: point the same client at Ollama serving Qwen and nothing in
the OS changes.

Airgap posture: binds to 127.0.0.1 only, never 0.0.0.0. Run it standalone:

    python -m agentos.inference.server            # serves NanoLM on 127.0.0.1:11434

then in another shell the default `auto` backend will discover and use it:

    python -m agentos run "Design a backup plan"
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from agentos.inference.backend import GenerationParams
from agentos.inference.nanolm import NanoLM

_MODEL = NanoLM()  # trained once, shared across worker threads (read-only)


class _Handler(BaseHTTPRequestHandler):
    server_version = "AgentOS-NanoLM/0.1"

    def log_message(self, *args) -> None:  # keep the console quiet
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": "nano", "object": "model"}]})
        else:
            self._send(200, {"status": "ok", "model": "nano"})

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": {"message": f"bad request: {exc}"}})
            return

        messages = req.get("messages", [])
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        params = GenerationParams(
            temperature=float(req.get("temperature", 0.7)),
            max_tokens=int(req.get("max_tokens", 64)),
            top_p=float(req.get("top_p", 0.95)),
        )
        text = _MODEL.complete(system, user, params)
        self._send(200, {
            "id": "nano-cmpl",
            "object": "chat.completion",
            "model": req.get("model", "nano"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(text.split()), "total_tokens": 0},
        })


def make_server(host: str = "127.0.0.1", port: int = 11434) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("airgap: the model server must bind to loopback only")
    return ThreadingHTTPServer((host, port), _Handler)


def serve_in_thread(host: str = "127.0.0.1", port: int = 0) -> Tuple[ThreadingHTTPServer, str]:
    """Start the server on a background thread; return (server, base_url).
    ``port=0`` picks a free port — handy for tests and demos."""
    server = make_server(host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    actual_port = server.server_address[1]
    return server, f"http://{host}:{actual_port}/v1"


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="agentos-serve", description="Local NanoLM inference server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11434)
    args = p.parse_args(argv)
    server = make_server(args.host, args.port)
    print(f"NanoLM serving OpenAI API on http://{args.host}:{args.port}/v1  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
