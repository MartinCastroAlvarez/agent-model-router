"""Security helpers: local auth token, backend URL validation, header allowlists, redaction.

Everything that decides what leaves the machine or what is written to disk lives here so
it can be reviewed in one place.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import re
import secrets
import stat
from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

__all__ = [
    "ConfigError",
    "check_backend_url",
    "is_loopback_host",
    "load_or_create_token",
    "redact",
    "tokens_match",
    "upstream_headers",
    "UPSTREAM_RESPONSE_HEADERS",
]


class ConfigError(ValueError):
    """Raised for an unsafe or invalid configuration."""


_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and loopback IP literals (``127.0.0.0/8``, ``::1``)."""
    host = host.strip("[]").lower()
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_backend_url(
    url: str, *, allow_insecure: bool = False, require_https: bool = False
) -> str:
    """Validate a backend base URL taken from the config file.

    Backend URLs only ever come from the operator's config, never from a request, so a
    client cannot point the router at an internal service. On top of that the URL must be
    plain ``http(s)://host[:port][/path]``: no credentials, query or fragment, and plain
    HTTP only for loopback hosts unless ``allow_insecure`` is set (a LAN vLLM box).
    Returns the URL without a trailing slash.
    """
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise ConfigError(f"backend url must be http(s): {url!r}")
    if not parts.hostname:
        raise ConfigError(f"backend url has no host: {url!r}")
    if parts.username or parts.password or "@" in parts.netloc:
        raise ConfigError("backend url must not embed credentials; use api_key_env")
    if parts.query or parts.fragment:
        raise ConfigError(f"backend url must not have a query or fragment: {url!r}")
    try:
        _ = parts.port
    except ValueError as exc:
        raise ConfigError(f"backend url has an invalid port: {url!r}") from exc
    loopback = is_loopback_host(parts.hostname)
    if parts.scheme == "http" and not loopback:
        if require_https:
            raise ConfigError(f"hosted backends must use https: {url!r}")
        if not allow_insecure:
            raise ConfigError(
                f"plain http to a non-loopback host needs allow_insecure: true: {url!r}"
            )
    return url.rstrip("/")


def tokens_match(presented: str | None, expected: str) -> bool:
    """Constant-time comparison of the client's token with the router's."""
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def load_or_create_token(env_var: str, token_file: Path) -> str:
    """Return the router's local auth token.

    Order: the environment variable, then the token file. When neither exists a random
    token is created in ``token_file`` with mode 0600. The token is never logged.
    """
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    token_file = token_file.expanduser()
    if token_file.exists():
        value = token_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    token_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    value = "amr-" + secrets.token_urlsafe(32)
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value + "\n")
    return value


# Inbound headers that may travel to a backend. Everything else (the client's
# Authorization / x-api-key, cookies, proxy headers, tracing ids) is dropped.
_FORWARD_SAME_API = {
    "anthropic": ("anthropic-version", "anthropic-beta"),
    "openai": ("openai-organization", "openai-project"),
}
_FORWARD_ANY = ("accept-language",)

# Upstream response headers that may travel back to the client.
UPSTREAM_RESPONSE_HEADERS = ("retry-after", "request-id", "x-request-id")


def upstream_headers(
    inbound: Mapping[str, str],
    *,
    inbound_api: str,
    backend_api: str,
    api_key: str | None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the headers sent to a backend from an allowlist.

    The router's own credentials for the backend come from the config (``api_key``);
    the client's credentials never leave the router.
    """
    out: dict[str, str] = {
        "content-type": "application/json",
        "user-agent": "agent-model-router",
    }
    lowered = {k.lower(): v for k, v in inbound.items()}
    names: Iterable[str] = _FORWARD_ANY
    if inbound_api == backend_api:
        names = (*names, *_FORWARD_SAME_API.get(backend_api, ()))
    for name in names:
        if name in lowered:
            out[name] = lowered[name]
    if backend_api == "anthropic":
        out.setdefault("anthropic-version", "2023-06-01")
        if api_key:
            out["x-api-key"] = api_key
    elif api_key:
        out["authorization"] = f"Bearer {api_key}"
    for name, value in (extra or {}).items():
        if name.lower() in {"authorization", "x-api-key", "cookie", "host"}:
            raise ConfigError(f"backend header {name!r} cannot be set statically")
        out[name.lower()] = value
    return out


_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"amr-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(password|passwd|secret|api[_-]?key|token)\b(\s*[:=]\s*)(['\"]?)[^\s'\"]{6,}\3"
    ),
]


def redact(text: str) -> tuple[str, bool]:
    """Replace secret-shaped substrings with ``[REDACTED]``; returns (text, changed)."""
    changed = False
    for pattern in _SECRET_PATTERNS:
        text, n = pattern.subn("[REDACTED]", text)
        changed = changed or n > 0
    return text, changed
