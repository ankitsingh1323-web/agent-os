"""Airgap enforcement.

The single most important invariant of this OS: model inference and every other
network call must stay on the local machine (or an explicitly trusted private
host). This module is the choke point that guarantees it. Backends call
``assert_local`` before opening any socket.
"""

from __future__ import annotations

import contextlib
import ipaddress
import socket
import threading
from dataclasses import dataclass, field
from urllib.parse import urlparse


class AirgapViolation(RuntimeError):
    """Raised when something tries to talk to a non-local endpoint."""


# Hostnames we treat as loopback without doing DNS (DNS itself is egress).
_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


@dataclass
class AirgapPolicy:
    """Controls what counts as 'local enough' to be allowed.

    Defaults are strict: only loopback and RFC-1918 private addresses. Operators
    can widen this to a trusted on-prem inference host, or disable enforcement
    entirely for a connected (non-airgapped) deployment — but that is an explicit,
    auditable choice, not the default.
    """

    enforce: bool = True
    allow_private_lan: bool = True
    extra_hosts: set = field(default_factory=set)

    def host_allowed(self, host: str) -> bool:
        host = (host or "").strip("[]")  # tolerate bracketed IPv6
        if host in _LOCAL_NAMES or host in self.extra_hosts:
            return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            # Unknown hostname: refuse. Resolving it would itself be egress.
            return False
        if ip.is_loopback:
            return True
        if self.allow_private_lan and (ip.is_private or ip.is_link_local):
            return True
        return False


# Process-wide default policy. Backends consult this unless handed another.
DEFAULT_POLICY = AirgapPolicy()


def assert_local(url: str, policy: AirgapPolicy | None = None) -> None:
    """Raise :class:`AirgapViolation` unless ``url`` points at an allowed host."""
    policy = policy or DEFAULT_POLICY
    if not policy.enforce:
        return
    host = urlparse(url).hostname or ""
    if not policy.host_allowed(host):
        raise AirgapViolation(
            f"Airgap policy blocked '{url}' (host={host!r}). "
            "Only loopback/private endpoints are permitted. "
            "Point this at a local model runtime (Ollama, vLLM, llama.cpp)."
        )


# --- process-level egress enforcement --------------------------------------
# ``assert_local`` protects the sanctioned inference path, but an externally-built
# agent might try its *own* network I/O and route around it. This guard closes
# that hole: while active, every non-local IP socket connect in the process is
# refused. That is what lets us run foreign agent code without breaking the airgap.

_guard_lock = threading.Lock()
_guard_depth = 0
_orig_connect = None


@contextlib.contextmanager
def process_egress_guard(policy: AirgapPolicy | None = None):
    """Block non-local socket connections for the duration of the block.

    Reentrant and concurrency-safe: the rule (allow local, refuse the rest) is
    uniform, so nested or concurrent guards compose. Local connections stay
    allowed, so the kernel's own model calls keep working. Non-IP sockets
    (e.g. AF_UNIX) are inherently local and pass through untouched.
    """
    global _guard_depth, _orig_connect
    policy = policy or DEFAULT_POLICY
    if not policy.enforce:
        yield
        return

    with _guard_lock:
        if _guard_depth == 0:
            _orig_connect = socket.socket.connect

            def _guarded_connect(self, address):
                if self.family in (socket.AF_INET, socket.AF_INET6):
                    host = address[0] if isinstance(address, tuple) else address
                    if not policy.host_allowed(str(host)):
                        raise AirgapViolation(
                            f"Airgap: blocked egress to {host!r} from within an "
                            "external agent. All I/O must go through kernel syscalls."
                        )
                return _orig_connect(self, address)

            socket.socket.connect = _guarded_connect
        _guard_depth += 1

    try:
        yield
    finally:
        with _guard_lock:
            _guard_depth -= 1
            if _guard_depth == 0 and _orig_connect is not None:
                socket.socket.connect = _orig_connect
