"""Config parsing and validation, including SSRF-safe backend URLs."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_model_router.config import ConfigError
from agent_model_router.config import load_config
from agent_model_router.config import parse_config
from tests.conftest import config_dict


def test_bundled_default_is_local_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_ROUTER_CONFIG", str(tmp_path / "missing.yaml"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = load_config()
    assert cfg.server.host == "127.0.0.1" and cfg.server.auth
    assert {m.backend for m in cfg.usable_models()} == {"ollama"}
    assert cfg.backends["ollama"].trusted and not cfg.backends["anthropic"].trusted
    assert cfg.baseline().name == "large"
    assert cfg.model("opus").effort_style == "adaptive"
    with pytest.raises(KeyError):
        cfg.model("nope")


def test_hosted_backend_needs_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AMR_TEST_ANTHROPIC_KEY", raising=False)
    cfg = parse_config(config_dict())
    assert "big" not in {m.name for m in cfg.usable_models()}
    monkeypatch.setenv("AMR_TEST_ANTHROPIC_KEY", "k")
    assert "big" in {m.name for m in cfg.usable_models()}
    assert cfg.backends["cloud"].api_key() == "k"
    assert cfg.backends["local"].api_key() is None


def test_load_from_path(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "backends: {o: {kind: ollama}}\nmodels: [{name: a, backend: o, model: m, "
        "quality: 0.5}]\nmetrics: {db: " + str(tmp_path / "x.db") + "}\n"
    )
    cfg = load_config(path)
    assert cfg.backends["o"].base_url == "http://127.0.0.1:11434/v1"
    assert cfg.metrics_db == str(tmp_path / "x.db")


@pytest.mark.parametrize(
    ("url", "extra", "message"),
    [
        ("ftp://host/v1", {}, r"http\(s\)"),
        ("http:///v1", {}, "no host"),
        ("http://user:pw@127.0.0.1/v1", {}, "credentials"),
        ("http://127.0.0.1/v1?x=1", {}, "query"),
        ("http://127.0.0.1:99999/v1", {}, "port"),
        ("http://10.0.0.5:8000/v1", {}, "allow_insecure"),
    ],
)
def test_bad_backend_urls(url: str, extra: dict, message: str) -> None:
    data = config_dict()
    data["backends"]["local"] = {"kind": "openai", "base_url": url, **extra}
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_hosted_backends_must_use_https() -> None:
    data = config_dict()
    data["backends"]["cloud"]["base_url"] = "http://api.anthropic.example"
    data["backends"]["cloud"]["allow_insecure"] = True
    with pytest.raises(ConfigError, match="https"):
        parse_config(data)


def test_insecure_lan_backend_is_allowed_explicitly_and_untrusted() -> None:
    cfg = parse_config(config_dict())
    assert cfg.backends["lan"].base_url == "http://gpu-box.lan:8000/v1"
    assert not cfg.backends["lan"].trusted


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(extra=1), "unknown keys"),
        (lambda d: d["backends"]["local"].update(kind="grpc"), "kind"),
        (lambda d: d["backends"]["local"].update(color="red"), "unknown keys"),
        (lambda d: d["backends"].update(x={"kind": "openai"}), "base_url is required"),
        (lambda d: d.update(backends={}), "at least one backend"),
        (lambda d: d.update(models=[]), "non-empty"),
        (lambda d: d["models"][0].pop("quality"), "needs 'quality'"),
        (lambda d: d["models"][0].update(backend="ghost"), "unknown backend"),
        (lambda d: d["models"][0].update(quality=2), "within"),
        (lambda d: d["models"][0].update(effort="turbo"), "effort"),
        (lambda d: d["models"].append(dict(d["models"][0])), "unique"),
        (lambda d: d["policy"].update(effort={"trivial": "max"}), "effort values"),
        (lambda d: d["policy"].update(quality={"giant": 1}), "keys must be"),
        (lambda d: d["policy"].update(sticky="maybe"), "sticky"),
        (lambda d: d["policy"].update(baseline_model="ghost"), "baseline_model"),
        (lambda d: d.update(classifier={"mode": "magic"}), "classifier.mode"),
        (lambda d: d.update(classifier={"mode": "decision"}), "decision_model"),
        (lambda d: d.update(budgets={"on_exceeded": "panic"}), "on_exceeded"),
        (lambda d: d.update(server={"host": "0.0.0.0"}), "allow_remote"),
        (
            lambda d: d.update(server={"host": "0.0.0.0", "allow_remote": True, "auth": False}),
            "auth: false",
        ),
        (lambda d: d.update(policy="x"), "mapping"),
        (lambda d: d["backends"]["lan"].update(headers={"Authorization": "x"}), None),
    ],
)
def test_invalid_configs(mutate, message: str | None) -> None:
    data = config_dict()
    mutate(data)
    if message is None:
        # static headers are validated when a request is built (see test_security)
        parse_config(data)
        return
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_remote_bind_with_opt_in_and_extras() -> None:
    data = config_dict(
        server={"host": "0.0.0.0", "allow_remote": True},
        classifier={"mode": "decision", "decision_model": "tiny"},
        budgets={"daily_usd": 2},
    )
    data["policy"]["baseline_model"] = "mid"
    data["policy"]["thinking_budgets"] = {"high": 30000}
    cfg = parse_config(data)
    assert cfg.server.host == "0.0.0.0" and cfg.classifier.decision_model == "tiny"
    assert cfg.budgets.daily_usd == 2.0 and cfg.baseline().name == "mid"
    assert cfg.policy.thinking_budgets["high"] == 30000


@pytest.mark.parametrize("name", ["claude-only.yaml", "mixed.yaml"])
def test_example_configs_parse(name: str) -> None:
    path = Path(__file__).resolve().parent.parent / "examples" / name
    cfg = load_config(path)
    assert cfg.models and cfg.server.host == "127.0.0.1"
