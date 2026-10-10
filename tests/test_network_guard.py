"""
Checks for the test-suite network guard in `tests/conftest.py`: external
connections and DNS lookups are blocked, localhost still works. No packet
leaves the machine - the external attempts are refused before any I/O.
"""

from __future__ import annotations

import asyncio
import socket
import urllib.request

import pytest


def test_external_dns_lookup_is_blocked(network_guard: list[str]) -> None:
    with pytest.raises(OSError, match="blocked in tests"):
        socket.getaddrinfo("serpapi.com", 443)

    assert network_guard == ["DNS lookup of serpapi.com"]
    network_guard.clear()  # expected attempt - don't fail this test at teardown


def test_external_connect_is_blocked(network_guard: list[str]) -> None:
    # 203.0.113.0/24 is a documentation-only range (TEST-NET-3), never routable.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(OSError, match="blocked"):
        sock.connect(("203.0.113.1", 443))

    assert network_guard == ["connect to 203.0.113.1"]
    network_guard.clear()


def test_urllib_request_to_external_host_is_blocked(network_guard: list[str]) -> None:
    with pytest.raises(OSError):
        urllib.request.urlopen("https://serpapi.com/search.json", timeout=5)

    assert network_guard == ["DNS lookup of serpapi.com"]
    network_guard.clear()


def test_localhost_connections_are_allowed() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass
    assert socket.getaddrinfo("localhost", None)


def test_asyncio_event_loop_still_works() -> None:
    # On Windows the event loop's internal socket pair connects over localhost.
    async def _noop() -> int:
        await asyncio.sleep(0)
        return 1

    assert asyncio.run(_noop()) == 1
