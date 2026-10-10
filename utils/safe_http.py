"""
Fetch untrusted URLs without letting them reach internal networks (SSRF guard).

URLs that come from search results are attacker-influenced: a listing can
point at `http://localhost:8501`, a private LAN address, or the cloud
metadata service (`169.254.169.254`), either directly or through a redirect.
`fetch()` refuses all of those.

Policy, applied to the initial URL and again to every redirect hop:
    - scheme must be http or https; no credentials in the URL; port 80/443 only
    - the host must be a real DNS name (letters, digits, hyphens, an
      alphabetic TLD): no IP literals in any notation, no single-label or
      `localhost`/`.local`/`.internal`-style names
    - at most `MAX_REDIRECTS` redirects
    - every address the host resolves to must be publicly routable. This is
      checked at connection time, on the exact addresses that are then
      connected to, so a DNS answer cannot change between check and connect.
      Loopback, private, link-local, CGNAT, multicast, reserved and
      metadata addresses all fail this check.
    - proxies from the environment are ignored: the address check would
      otherwise apply to the proxy, not to the real destination.
    - the response body is capped at `max_bytes` and the whole transfer at
      `total_timeout` seconds, not only each socket operation.

Raises `UnsafeURLError` (a `ValueError`) for policy violations; callers that
treat fetching as best-effort can catch `Exception`.
"""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Final

MAX_REDIRECTS: Final[int] = 3

_ALLOWED_PORTS: Final[frozenset[int]] = frozenset({80, 443})
_READ_CHUNK_BYTES: Final[int] = 16_384

# A DNS name: dot-separated labels with an alphabetic TLD. Anything numeric
# ("2130706433", "0x7f.1", "127.1") or single-label ("intranet") is rejected.
_HOSTNAME = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_INTERNAL_SUFFIXES: Final[tuple[str, ...]] = (
    ".localhost",
    ".local",
    ".localdomain",
    ".internal",
    ".intranet",
    ".lan",
    ".home",
    ".corp",
    ".home.arpa",
)

_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


class UnsafeURLError(ValueError):
    """The URL (or a redirect, or an address it resolves to) is not allowed to be fetched."""


def is_public_ip(address: str) -> bool:
    """True only for a globally routable unicast address."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        # Addresses that merely wrap an IPv4 address are only as public as that IPv4 address.
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip.teredo is not None:
            embedded = ip.teredo[1]
        if embedded is None and ip in _NAT64_PREFIX:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            return is_public_ip(str(embedded))
    return ip.is_global and not ip.is_multicast


def check_url(url: str) -> urllib.parse.SplitResult:
    """Validate `url` against the policy above (no network access). Returns the parsed URL."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError(f"malformed URL: {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError(f"scheme {parts.scheme!r} is not allowed")
    if parts.username is not None or parts.password is not None:
        raise UnsafeURLError("credentials in the URL are not allowed")
    if port is not None and port not in _ALLOWED_PORTS:
        raise UnsafeURLError(f"port {port} is not allowed")

    host = (parts.hostname or "").rstrip(".")
    try:
        host = host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise UnsafeURLError("invalid host name") from exc
    if not _HOSTNAME.match(host):
        raise UnsafeURLError(f"host {host!r} is not a public DNS name")
    if host == "localhost" or host.endswith(_INTERNAL_SUFFIXES):
        raise UnsafeURLError(f"host {host!r} is an internal name")
    return parts


def _resolve_public(host: str, port: int) -> list[tuple[Any, ...]]:
    """DNS-resolve `host`; every returned address must be public or the whole host is refused."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not infos:
        raise UnsafeURLError(f"{host!r} did not resolve")
    for _family, _type, _proto, _canon, sockaddr in infos:
        if not is_public_ip(str(sockaddr[0])):
            raise UnsafeURLError(f"{host!r} resolves to a non-public address")
    return list(infos)


def _connect(infos: list[tuple[Any, ...]], timeout: float | None) -> socket.socket:
    """Connect to the first reachable already-vetted address."""
    last_error: OSError = OSError("no addresses to connect to")
    for family, socktype, proto, _canon, sockaddr in infos:
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            sock.connect(sockaddr)
        except OSError as exc:
            sock.close()
            last_error = exc
        else:
            return sock
    raise last_error


def _open_checked_socket(host: str, port: int, timeout: float | None) -> socket.socket:
    return _connect(_resolve_public(host, port), timeout)


class _SafeHTTPConnection(http.client.HTTPConnection):
    def connect(self) -> None:
        self.sock = _open_checked_socket(self.host, self.port, self.timeout)


class _SafeHTTPSConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        sock = _open_checked_socket(self.host, self.port, self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)  # type: ignore[attr-defined]


class _SafeHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_SafeHTTPConnection, req)


class _SafeHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_SafeHTTPSConnection, req, context=self._context)  # type: ignore[attr-defined]


class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        check_url(urllib.parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        _SafeHTTPHandler(),
        _SafeHTTPSHandler(context=ssl.create_default_context()),
        _CheckedRedirectHandler(),
        urllib.request.ProxyHandler({}),  # explicitly no environment proxies
    )


@dataclass(frozen=True)
class FetchedResponse:
    body: bytes
    content_type: str
    charset: str | None


def fetch(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 8.0,
    total_timeout: float = 15.0,
    max_bytes: int = 400_000,
    require_content_type: str | None = None,
) -> FetchedResponse:
    """
    GET `url` under the policy in the module docstring. At most `max_bytes`
    of the body are read (the rest is simply not downloaded). If
    `require_content_type` is given, the response's content type must contain
    it (e.g. "html"), checked before the body is read.
    """
    check_url(url)
    deadline = time.monotonic() + total_timeout
    request = urllib.request.Request(url, headers=headers or {})
    with _build_opener().open(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        if require_content_type and require_content_type not in content_type:
            raise ValueError(f"unexpected content type {content_type}")
        chunks: list[bytes] = []
        remaining = max_bytes
        while remaining > 0:
            if time.monotonic() > deadline:
                raise TimeoutError(f"transfer exceeded {total_timeout:g}s")
            chunk = response.read(min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return FetchedResponse(
            b"".join(chunks), content_type, response.headers.get_content_charset()
        )
