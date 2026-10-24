"""CLI commands (serve is checked without binding a socket)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_model_router import cli
from agent_model_router.metrics import MetricsStore
from agent_model_router.metrics import RouteRecord
from tests.conftest import HARD
from tests.conftest import TRIVIAL
from tests.conftest import config_dict


@pytest.fixture
def cfg_path(tmp_path: Path, anthropic_key: str) -> Path:
    data = config_dict(
        metrics={"db": str(tmp_path / "routes.sqlite3")},
        server={"key_env": "AMR_CLI_KEY", "key_file": str(tmp_path / "token")},
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = cli.main(list(args))
    out, err = capsys.readouterr()
    return code, out, err


def test_init_writes_the_default(tmp_path: Path, capsys) -> None:
    target = tmp_path / "c.yaml"
    assert run(capsys, "init", str(target))[0] == 0
    assert "backends:" in target.read_text()
    code, _, err = run(capsys, "init", str(target))
    assert code == 1 and "exists" in err
    assert run(capsys, "init", str(target), "--force")[0] == 0


def test_token(cfg_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("AMR_CLI_KEY", raising=False)
    code, out, _ = run(capsys, "--config", str(cfg_path), "token")
    assert code == 0 and out.strip().startswith("amr-")
    assert run(capsys, "--config", str(cfg_path), "token")[1] == out


def test_explain_plain_text_and_json(cfg_path: Path, tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(TRIVIAL))
    code, out, _ = run(capsys, "--config", str(cfg_path), "explain", "-")
    assert code == 0 and json.loads(out)["model"] == "tiny"
    body = tmp_path / "b.json"
    body.write_text(json.dumps({"messages": [{"role": "user", "content": HARD}]}))
    code, out, _ = run(
        capsys, "--config", str(cfg_path), "explain", str(body), "--api", "openai", "--offline"
    )
    result = json.loads(out)
    assert result["model"] == "big" and result["baseline_model"] == "big"


def test_explain_without_models(tmp_path: Path, capsys) -> None:
    data = config_dict()
    for b in data["backends"].values():
        b["enabled"] = False
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(data))
    prompt = tmp_path / "p.txt"
    prompt.write_text("hello")
    code, out, _ = run(capsys, "--config", str(path), "explain", str(prompt))
    assert code == 1 and "error" in json.loads(out)


def _record(**kw: Any) -> RouteRecord:
    base: dict[str, Any] = dict(
        session="s",
        api="anthropic",
        task="edit",
        bucket="trivial",
        difficulty=0.1,
        model="tiny",
        effort="none",
        reason="r",
        status=200,
        input_tokens=1000,
        output_tokens=100,
        estimated=False,
        cost_usd=0.00001,
        baseline_usd=0.003,
        latency_ms=10.0,
        attempts=1,
        stream=False,
    )
    base.update(kw)
    return RouteRecord(**base)


def test_stats(cfg_path: Path, tmp_path: Path, capsys) -> None:
    code, _, err = run(capsys, "--config", str(cfg_path), "stats")
    assert code == 1 and "no route log" in err
    store = MetricsStore(tmp_path / "routes.sqlite3")
    store.add(_record())
    store.add(_record(model="big", cost_usd=0.003, attempts=2, status=502))
    store.close()
    code, out, _ = run(capsys, "--config", str(cfg_path), "stats")
    assert code == 0 and "tiny" in out and "saved:" in out and "always-big" in out
    code, out, _ = run(capsys, "--config", str(cfg_path), "stats", "--json", "--hours", "1")
    summary = json.loads(out)
    assert summary["requests"] == 2 and round(summary["savings_pct"]) == 50
    assert {m["model"]: m["errors"] for m in summary["models"]} == {"tiny": 0, "big": 1}


def test_stats_disabled(tmp_path: Path, capsys) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config_dict()))
    code, _, err = run(capsys, "--config", str(path), "stats")
    assert code == 1 and "disabled" in err


def test_train_and_eval(tmp_path: Path, capsys) -> None:
    from agent_model_router.classifier.dataset import load_rows

    data = tmp_path / "small.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in load_rows()[:300]))
    out_path = tmp_path / "w.json"
    code, out, _ = run(capsys, "train", "--data", str(data), "--out", str(out_path))
    report = json.loads(out)
    assert code == 0 and report["train"]["rows"] == 300
    assert report["eval"]["bucket_within_one"] >= 0.8
    code, out, _ = run(capsys, "eval", "--weights", str(out_path))
    assert code == 0 and json.loads(out)["rows"] > 20


def test_config_errors_exit_2(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("models: []\nbackends: {a: {kind: ollama}}\n")
    code, _, err = run(capsys, "--config", str(path), "token")
    assert code == 2 and "config error" in err


def test_serve(cfg_path: Path, capsys, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.append(kw))
    monkeypatch.setenv("AMR_CLI_KEY", "amr-secret-from-env")
    code, _, err = run(capsys, "--config", str(cfg_path), "serve", "--port", "9999")
    assert code == 0 and calls[0]["host"] == "127.0.0.1" and calls[0]["port"] == 9999
    assert calls[0]["access_log"] is False
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:9999" in err
    assert "amr-secret-from-env" not in err  # the token is never printed by serve
    code, _, err = run(capsys, "--config", str(cfg_path), "serve", "--host", "0.0.0.0")
    assert code == 2 and "non-loopback" in err
    assert len(calls) == 1
