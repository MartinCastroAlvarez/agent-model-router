"""Router configuration: backends, the model pool, policy, budgets.

The config is YAML read with ``yaml.safe_load`` and validated into frozen dataclasses.
Backend URLs are validated here (see :func:`security.check_backend_url`); nothing in a
request can add or change a backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from agent_model_router.security import ConfigError
from agent_model_router.security import check_backend_url
from agent_model_router.security import is_loopback_host

__all__ = [
    "BUCKETS",
    "EFFORTS",
    "Backend",
    "Budgets",
    "ClassifierConfig",
    "Config",
    "ConfigError",
    "Model",
    "Policy",
    "Server",
    "default_config_path",
    "load_config",
]

BUCKETS = ("trivial", "easy", "medium", "hard", "expert")
EFFORTS = ("none", "low", "medium", "high")
EFFORT_STYLES = ("none", "reasoning_effort", "adaptive", "budget")
BACKEND_KINDS = ("ollama", "openai", "anthropic")
HOSTED_KINDS = ("anthropic",)

_DEFAULT_BASE_URLS = {
    "ollama": "http://127.0.0.1:11434/v1",
    "anthropic": "https://api.anthropic.com",
}


@dataclass(frozen=True)
class Backend:
    """An upstream server. ``api`` is the wire protocol the router speaks to it."""

    name: str
    kind: str
    base_url: str
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 600.0
    connect_timeout_s: float = 5.0
    enabled: bool = True
    trusted: bool = False  # may receive prompts that contain secrets (default: loopback only)

    @property
    def api(self) -> str:
        return "anthropic" if self.kind == "anthropic" else "openai"

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env, "").strip()
        return value or None

    @property
    def available(self) -> bool:
        """Enabled, and its key is present when it needs one."""
        return self.enabled and (self.api_key_env is None or self.api_key() is not None)


@dataclass(frozen=True)
class Model:
    """One entry of the pool: a backend model with a quality score and a price."""

    name: str
    backend: str
    model: str
    quality: float
    price_in: float = 0.0  # USD per million input tokens (notional for local models)
    price_out: float = 0.0  # USD per million output tokens
    context: int = 32768
    max_output: int = 8192
    tools: bool = True
    effort_style: str = "none"
    enabled: bool = True


@dataclass(frozen=True)
class Policy:
    quality: dict[str, float]
    effort: dict[str, str]
    sticky: str = "ratchet"  # ratchet | off
    idle_reset_minutes: float = 30.0
    max_attempts: int = 3
    escalate_on_refusal: bool = True
    baseline_model: str | None = None
    thinking_budgets: dict[str, int] = field(
        default_factory=lambda: {"low": 2048, "medium": 8192, "high": 24576}
    )


@dataclass(frozen=True)
class ClassifierConfig:
    mode: str = "heuristic"  # heuristic | decision
    decision_model: str | None = None
    decision_timeout_s: float = 3.0
    weights: str | None = None  # path to custom weights JSON


@dataclass(frozen=True)
class Budgets:
    daily_usd: float | None = None
    on_exceeded: str = "cheapest"  # cheapest | reject


@dataclass(frozen=True)
class Server:
    host: str = "127.0.0.1"
    port: int = 8787
    auth: bool = True
    key_env: str = "AGENT_ROUTER_TOKEN"
    key_file: str = "~/.config/agent-model-router/token"
    allow_remote: bool = False
    allowed_origins: tuple[str, ...] = ()
    max_body_bytes: int = 32 * 1024 * 1024


@dataclass(frozen=True)
class Config:
    backends: dict[str, Backend]
    models: tuple[Model, ...]
    policy: Policy
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    budgets: Budgets = field(default_factory=Budgets)
    server: Server = field(default_factory=Server)
    metrics_db: str | None = "~/.local/share/agent-model-router/routes.sqlite3"

    def model(self, name: str) -> Model:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)

    def usable_models(self) -> list[Model]:
        """Enabled models whose backend is available (enabled, key present)."""
        return [m for m in self.models if m.enabled and self.backends[m.backend].available]

    def baseline(self) -> Model:
        """The model savings are measured against: configured, else the priciest usable one."""
        if self.policy.baseline_model:
            return self.model(self.policy.baseline_model)
        pool = self.usable_models() or list(self.models)
        return max(pool, key=lambda m: (m.price_in + m.price_out, m.quality))


def default_config_path() -> Path:
    env = os.environ.get("AGENT_ROUTER_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path("~/.config/agent-model-router/config.yaml").expanduser()


def _bundled_config_text() -> str:
    return (
        resources.files("agent_model_router")
        .joinpath("data/default.yaml")
        .read_text(encoding="utf-8")
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load the config from ``path``, ``$AGENT_ROUTER_CONFIG``, the user file or the default."""
    if path is None:
        candidate = default_config_path()
        text = (
            candidate.read_text(encoding="utf-8") if candidate.exists() else _bundled_config_text()
        )
    else:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return parse_config(data)


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _unknown(data: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(data) - allowed
    if extra:
        raise ConfigError(f"unknown keys in {where}: {sorted(extra)}")


def _parse_backend(name: str, raw: dict[str, Any]) -> Backend:
    _unknown(
        raw,
        {
            "kind",
            "base_url",
            "api_key_env",
            "headers",
            "timeout_s",
            "connect_timeout_s",
            "enabled",
            "allow_insecure",
            "trusted",
        },
        f"backends.{name}",
    )
    kind = raw.get("kind", "openai")
    if kind not in BACKEND_KINDS:
        raise ConfigError(f"backends.{name}.kind must be one of {BACKEND_KINDS}")
    url = raw.get("base_url") or _DEFAULT_BASE_URLS.get(kind)
    if not url:
        raise ConfigError(f"backends.{name}.base_url is required for kind {kind!r}")
    url = check_backend_url(
        str(url),
        allow_insecure=bool(raw.get("allow_insecure", False)),
        require_https=kind in HOSTED_KINDS,
    )
    headers = {str(k): str(v) for k, v in _mapping(raw.get("headers"), "headers").items()}
    return Backend(
        name=name,
        kind=kind,
        base_url=url,
        api_key_env=raw.get("api_key_env"),
        headers=headers,
        timeout_s=float(raw.get("timeout_s", 600.0)),
        connect_timeout_s=float(raw.get("connect_timeout_s", 5.0)),
        enabled=bool(raw.get("enabled", True)),
        trusted=bool(raw.get("trusted", is_loopback_host(urlsplit(url).hostname or ""))),
    )


def _parse_model(raw: dict[str, Any], backends: dict[str, Backend]) -> Model:
    _unknown(
        raw,
        {
            "name",
            "backend",
            "model",
            "quality",
            "price",
            "context",
            "max_output",
            "tools",
            "effort",
            "enabled",
        },
        f"models[{raw.get('name')}]",
    )
    for key in ("name", "backend", "model", "quality"):
        if key not in raw:
            raise ConfigError(f"every model needs {key!r}")
    if raw["backend"] not in backends:
        raise ConfigError(f"model {raw['name']!r} uses unknown backend {raw['backend']!r}")
    quality = float(raw["quality"])
    if not 0.0 <= quality <= 1.0:
        raise ConfigError(f"model {raw['name']!r}: quality must be within [0, 1]")
    price = _mapping(raw.get("price"), "price")
    style = raw.get("effort", "none")
    if style not in EFFORT_STYLES:
        raise ConfigError(f"model {raw['name']!r}: effort must be one of {EFFORT_STYLES}")
    return Model(
        name=str(raw["name"]),
        backend=str(raw["backend"]),
        model=str(raw["model"]),
        quality=quality,
        price_in=float(price.get("input", 0.0)),
        price_out=float(price.get("output", 0.0)),
        context=int(raw.get("context", 32768)),
        max_output=int(raw.get("max_output", 8192)),
        tools=bool(raw.get("tools", True)),
        effort_style=style,
        enabled=bool(raw.get("enabled", True)),
    )


def parse_config(data: Any) -> Config:
    """Validate a decoded YAML document into a :class:`Config`."""
    data = _mapping(data, "config")
    _unknown(
        data,
        {"server", "backends", "models", "policy", "classifier", "budgets", "metrics"},
        "config",
    )
    backends = {
        name: _parse_backend(name, _mapping(raw, f"backends.{name}"))
        for name, raw in _mapping(data.get("backends"), "backends").items()
    }
    if not backends:
        raise ConfigError("at least one backend is required")
    models_raw = data.get("models") or []
    if not isinstance(models_raw, list) or not models_raw:
        raise ConfigError("models must be a non-empty list")
    models = tuple(_parse_model(_mapping(m, "model"), backends) for m in models_raw)
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ConfigError("model names must be unique")

    p = _mapping(data.get("policy"), "policy")
    _unknown(
        p,
        {
            "quality",
            "effort",
            "sticky",
            "idle_reset_minutes",
            "max_attempts",
            "escalate_on_refusal",
            "baseline_model",
            "thinking_budgets",
        },
        "policy",
    )
    quality = {"trivial": 0.0, "easy": 0.3, "medium": 0.55, "hard": 0.75, "expert": 0.9}
    quality.update({k: float(v) for k, v in _mapping(p.get("quality"), "quality").items()})
    effort = {
        "trivial": "none",
        "easy": "low",
        "medium": "medium",
        "hard": "high",
        "expert": "high",
    }
    effort.update({k: str(v) for k, v in _mapping(p.get("effort"), "effort").items()})
    if set(quality) != set(BUCKETS) or set(effort) != set(BUCKETS):
        raise ConfigError(f"policy.quality/effort keys must be {BUCKETS}")
    if any(v not in EFFORTS for v in effort.values()):
        raise ConfigError(f"policy.effort values must be one of {EFFORTS}")
    sticky = p.get("sticky", "ratchet")
    if sticky not in {"ratchet", "off"}:
        raise ConfigError("policy.sticky must be 'ratchet' or 'off'")
    baseline = p.get("baseline_model")
    if baseline is not None and baseline not in names:
        raise ConfigError(f"policy.baseline_model {baseline!r} is not in models")
    budgets_tokens = {"low": 2048, "medium": 8192, "high": 24576}
    budgets_tokens.update(
        {k: int(v) for k, v in _mapping(p.get("thinking_budgets"), "thinking_budgets").items()}
    )
    policy = Policy(
        quality=quality,
        effort=effort,
        sticky=sticky,
        idle_reset_minutes=float(p.get("idle_reset_minutes", 30.0)),
        max_attempts=max(1, int(p.get("max_attempts", 3))),
        escalate_on_refusal=bool(p.get("escalate_on_refusal", True)),
        baseline_model=baseline,
        thinking_budgets=budgets_tokens,
    )

    c = _mapping(data.get("classifier"), "classifier")
    _unknown(c, {"mode", "decision_model", "decision_timeout_s", "weights"}, "classifier")
    mode = c.get("mode", "heuristic")
    if mode not in {"heuristic", "decision"}:
        raise ConfigError("classifier.mode must be 'heuristic' or 'decision'")
    decision_model = c.get("decision_model")
    if mode == "decision" and decision_model not in names:
        raise ConfigError("classifier.decision_model must name a model in the pool")
    classifier = ClassifierConfig(
        mode=mode,
        decision_model=decision_model,
        decision_timeout_s=float(c.get("decision_timeout_s", 3.0)),
        weights=c.get("weights"),
    )

    b = _mapping(data.get("budgets"), "budgets")
    _unknown(b, {"daily_usd", "on_exceeded"}, "budgets")
    on_exceeded = b.get("on_exceeded", "cheapest")
    if on_exceeded not in {"cheapest", "reject"}:
        raise ConfigError("budgets.on_exceeded must be 'cheapest' or 'reject'")
    daily = b.get("daily_usd")
    budgets = Budgets(daily_usd=None if daily is None else float(daily), on_exceeded=on_exceeded)

    s = _mapping(data.get("server"), "server")
    _unknown(
        s,
        {
            "host",
            "port",
            "auth",
            "key_env",
            "key_file",
            "allow_remote",
            "allowed_origins",
            "max_body_bytes",
        },
        "server",
    )
    server = Server(
        host=str(s.get("host", "127.0.0.1")),
        port=int(s.get("port", 8787)),
        auth=bool(s.get("auth", True)),
        key_env=str(s.get("key_env", "AGENT_ROUTER_TOKEN")),
        key_file=str(s.get("key_file", "~/.config/agent-model-router/token")),
        allow_remote=bool(s.get("allow_remote", False)),
        allowed_origins=tuple(str(o) for o in s.get("allowed_origins") or ()),
        max_body_bytes=int(s.get("max_body_bytes", 32 * 1024 * 1024)),
    )
    if not is_loopback_host(server.host) and not server.allow_remote:
        raise ConfigError("server.host is not loopback; set allow_remote: true to expose it")
    if not server.auth and not is_loopback_host(server.host):
        raise ConfigError("server.auth: false is only allowed on a loopback host")

    m = _mapping(data.get("metrics"), "metrics")
    _unknown(m, {"db", "enabled"}, "metrics")
    metrics_db: str | None = str(m.get("db", "~/.local/share/agent-model-router/routes.sqlite3"))
    if m.get("enabled") is False:
        metrics_db = None
    return Config(
        backends=backends,
        models=models,
        policy=policy,
        classifier=classifier,
        budgets=budgets,
        server=server,
        metrics_db=metrics_db,
    )
