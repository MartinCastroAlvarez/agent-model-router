"""Security properties: tokens, header allowlists, redaction, and a source-level audit."""

from __future__ import annotations

import ast
import os
import stat
from pathlib import Path

import pytest

from agent_model_router.security import ConfigError
from agent_model_router.security import is_loopback_host
from agent_model_router.security import load_or_create_token
from agent_model_router.security import redact
from agent_model_router.security import tokens_match
from agent_model_router.security import upstream_headers

PACKAGE = Path(__file__).resolve().parent.parent / "agent_model_router"


def test_loopback_detection() -> None:
    for host in ("localhost", "127.0.0.1", "127.8.9.1", "::1", "[::1]"):
        assert is_loopback_host(host)
    for host in ("0.0.0.0", "10.0.0.1", "evil.example", "localhost.evil.example", ""):
        assert not is_loopback_host(host)


def test_tokens_match() -> None:
    assert tokens_match("abc", "abc")
    assert not tokens_match("abd", "abc")
    assert not tokens_match(None, "abc")
    assert not tokens_match("", "abc")


def test_token_file_is_created_private_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AMR_KEY", raising=False)
    path = tmp_path / "dir" / "token"
    first = load_or_create_token("AMR_KEY", path)
    assert first.startswith("amr-") and len(first) > 40
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert load_or_create_token("AMR_KEY", path) == first
    monkeypatch.setenv("AMR_KEY", "from-env")
    assert load_or_create_token("AMR_KEY", path) == "from-env"
    monkeypatch.delenv("AMR_KEY")
    path.write_text("\n")
    assert load_or_create_token("AMR_KEY", path) != first  # empty file -> regenerated


def test_upstream_headers_are_an_allowlist() -> None:
    inbound = {
        "Authorization": "Bearer client-token",
        "x-api-key": "client-token",
        "Cookie": "a=b",
        "anthropic-beta": "x",
        "anthropic-version": "2023-06-01",
        "accept-language": "es",
        "x-forwarded-for": "1.1.1.1",
        "openai-organization": "org",
        "host": "127.0.0.1:8787",
    }
    out = upstream_headers(
        inbound, inbound_api="anthropic", backend_api="anthropic", api_key="router-key"
    )
    assert out["x-api-key"] == "router-key" and "authorization" not in out
    assert out["anthropic-beta"] == "x" and out["accept-language"] == "es"
    assert "cookie" not in out and "x-forwarded-for" not in out and "host" not in out
    out = upstream_headers(inbound, inbound_api="openai", backend_api="anthropic", api_key=None)
    assert "anthropic-beta" not in out and out["anthropic-version"] == "2023-06-01"
    assert "x-api-key" not in out
    out = upstream_headers(
        inbound,
        inbound_api="openai",
        backend_api="openai",
        api_key="k",
        extra={"HTTP-Referer": "https://example.org"},
    )
    assert out["authorization"] == "Bearer k" and out["openai-organization"] == "org"
    assert out["http-referer"] == "https://example.org"
    assert "client-token" not in str(out)
    for name in ("Authorization", "x-api-key", "Cookie", "Host"):
        with pytest.raises(ConfigError):
            upstream_headers(
                {}, inbound_api="openai", backend_api="openai", api_key=None, extra={name: "x"}
            )


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-abcdefghijklmnop",
        "sk-proj-abcdefghijklmnopqrstu",
        "ghp_abcdefghijklmnopqrstuvwxyz012345",
        "AKIAABCDEFGHIJKLMNOP",
        "xoxb-1234567890-abcdef",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijk",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----",
        "password = hunter2hunter2",
        "API_KEY: 'abcdef123456'",
    ],
)
def test_redact(secret: str) -> None:
    text, changed = redact(f"before {secret} after")
    assert changed and "[REDACTED]" in text
    assert text.startswith("before") and text.endswith("after")


def test_redact_leaves_normal_text() -> None:
    assert redact("rename token_count to n_tokens") == ("rename token_count to n_tokens", False)


def _sources() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_no_dangerous_calls_in_the_package() -> None:
    banned_calls = {"eval", "exec", "compile", "__import__"}
    banned_modules = {"pickle", "marshal", "subprocess", "shelve", "dill"}
    for path in _sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in banned_calls, f"{path}: {node.func.id}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"load", "unsafe_load"} or not (
                    isinstance(node.func.value, ast.Name) and node.func.value.id == "yaml"
                ), f"{path}: yaml.{node.func.attr}"
                for kw in node.keywords:
                    assert not (kw.arg == "shell" and getattr(kw.value, "value", False)), path
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    assert name.split(".")[0] not in banned_modules, f"{path}: {name}"


def test_no_suppression_comments_in_the_package() -> None:
    for path in _sources():
        text = path.read_text()
        for marker in ("# noqa", "# nosec", "# type: ignore"):
            assert marker not in text, f"{path}: {marker}"


def test_http_clients_never_follow_redirects_or_proxy_env(config) -> None:
    from agent_model_router.backends import Upstream

    upstream = Upstream(config)
    for client in upstream.clients.values():
        assert client.follow_redirects is False
        assert client.trust_env is False
