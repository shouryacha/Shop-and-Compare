"""
Network guard for the whole test suite: no test may reach the network.

Every test runs with `socket.socket.connect` and `socket.getaddrinfo`
replaced, so any connection or DNS lookup to a non-localhost host raises
`OSError` before a packet leaves the machine. This protects the real
SerpApi / NVIDIA keys in `.env` from being used by a test that forgets to
mock something (which previously happened in the Shop & Compare tests).

Because much of the production code deliberately catches network errors
and carries on, a blocked attempt could otherwise pass silently - so any
test that attempted network access is also FAILED at teardown, naming the
host.

Localhost (127.0.0.1, ::1, "localhost") and non-IP sockets stay allowed.
Tests that mock at a higher level (e.g. `_fetch_json`, `_classify`) never
reach the socket layer and are unaffected.

A test that deliberately checks the guard can request the `network_guard`
fixture to inspect the recorded attempts, and clear them to show they
were expected.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_loopback(host: object) -> bool:
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    return str(host).lower() in _LOOPBACK_HOSTS


@pytest.fixture(autouse=True)
def network_guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock: socket.socket, address: Any) -> None:
        if not isinstance(address, tuple) or _is_loopback(address[0]):
            return real_connect(sock, address)
        attempts.append(f"connect to {address[0]}")
        raise OSError(f"Network access is blocked in tests (connect to {address[0]})")

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is None or _is_loopback(host):
            return real_getaddrinfo(host, *args, **kwargs)
        attempts.append(f"DNS lookup of {host}")
        raise OSError(f"Network access is blocked in tests (DNS lookup of {host})")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    yield attempts

    if attempts:
        pytest.fail(
            "Test attempted network access (blocked): " + ", ".join(attempts),
            pytrace=False,
        )
