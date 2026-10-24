"""Policy: cheapest model that clears the bar, filters, stickiness, chains, budgets."""

from __future__ import annotations

import pytest

from agent_model_router.classifier.heuristic import Decision
from agent_model_router.config import Config
from agent_model_router.config import parse_config
from agent_model_router.policy import NoRoute
from agent_model_router.policy import SessionState
from agent_model_router.policy import SessionStore
from agent_model_router.policy import record
from agent_model_router.policy import select
from agent_model_router.view import view_request
from tests.conftest import anthropic_body
from tests.conftest import config_dict


def decision(bucket: str, task: str = "edit", secret: bool = False) -> Decision:
    return Decision(task=task, difficulty=0.5, bucket=bucket, source="test", secret=secret)


def route(config: Config, bucket: str, *, body=None, session=None, **kw):
    v = view_request("anthropic", body or anthropic_body("x"))
    return select(
        config,
        decision(bucket, secret=kw.pop("secret", False)),
        v,
        session_key="k",
        session=session,
        **kw,
    )


def test_cheapest_model_that_clears_the_bar(config: Config) -> None:
    assert route(config, "trivial").model.name == "tiny"
    assert route(config, "easy").model.name == "tiny"
    assert route(config, "medium").model.name == "mid"
    assert route(config, "hard").model.name == "big"
    r = route(config, "expert")
    assert r.model.name == "big" and r.effort == "high"


def test_none_clears_uses_the_best(config: Config) -> None:
    data = config_dict()
    data["policy"]["quality"]["expert"] = 0.99
    r = route(parse_config(data), "expert")
    assert r.model.name == "big" and "no model clears" in r.reason


def test_effort_follows_the_bucket(config: Config) -> None:
    assert [route(config, b).effort for b in ("trivial", "easy", "medium", "hard")] == [
        "none",
        "low",
        "medium",
        "high",
    ]


def test_tools_and_context_filters() -> None:
    data = config_dict()
    data["models"][0]["tools"] = False
    cfg = parse_config(data)
    body = anthropic_body("x", tools=[{"name": "f", "input_schema": {}}])
    r = route(cfg, "trivial", body=body)
    assert r.model.name == "mid"
    assert next(c for c in r.candidates if c["model"] == "tiny")["excluded"] == "no tool support"
    big_context = anthropic_body("y" * 4 * 60000)
    r = route(cfg, "trivial", body=big_context)
    assert r.model.name == "mid"
    assert "context" in str(next(c for c in r.candidates if c["model"] == "tiny")["excluded"])
    with pytest.raises(NoRoute):
        route(cfg, "trivial", body=anthropic_body("z" * 4 * 1_100_000))


def test_no_usable_models() -> None:
    data = config_dict()
    for b in data["backends"].values():
        b["enabled"] = False
    with pytest.raises(NoRoute):
        route(parse_config(data), "easy")


def test_chain_goes_up_then_down_and_respects_max_attempts(config: Config) -> None:
    assert [m.name for m in route(config, "trivial").chain] == ["tiny", "mid", "big"]
    assert [m.name for m in route(config, "medium").chain] == ["mid", "big", "tiny"]
    data = config_dict()
    data["policy"]["max_attempts"] = 2
    assert [m.name for m in route(parse_config(data), "hard").chain] == ["big", "mid"]


def test_secret_restricts_to_trusted_backends(config: Config) -> None:
    r = route(config, "expert", secret=True)
    assert r.model.name == "tiny" and [m.name for m in r.chain] == ["tiny"]


def test_budget_and_pin(config: Config) -> None:
    r = route(config, "expert", budget_exceeded=True)
    assert r.model.name == "tiny" and "budget" in r.reason
    r = route(config, "trivial", pin="big")
    assert r.model.name == "big" and "pinned" in r.reason
    assert route(config, "trivial", pin="nope").model.name == "tiny"


def test_sticky_ratchet(config: Config) -> None:
    on_big = SessionState(model="big", quality=0.9, last_seen=0)
    r = route(config, "trivial", session=on_big)
    assert r.model.name == "big" and r.sticky and "ratchet" in r.reason
    on_tiny = SessionState(model="tiny", quality=0.3, last_seen=0)
    r = route(config, "hard", session=on_tiny)
    assert r.model.name == "big" and "up from tiny" in r.reason
    same = SessionState(model="tiny", quality=0.3, last_seen=0)
    r = route(config, "trivial", session=same)
    assert r.model.name == "tiny" and not r.sticky
    gone = SessionState(model="removed", quality=0.5, last_seen=0)
    assert route(config, "trivial", session=gone).model.name == "tiny"


def test_escalate_after_refusal(config: Config) -> None:
    refused = SessionState(model="tiny", quality=0.3, last_seen=0, escalate_next=True)
    r = route(config, "trivial", session=refused)
    assert r.model.name == "mid" and "refusal" in r.reason
    top = SessionState(model="big", quality=0.9, last_seen=0, escalate_next=True)
    r = route(config, "trivial", session=top)
    assert r.model.name == "big" and r.sticky


def test_tool_followup_keeps_the_model(config: Config) -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "1", "content": "x"}],
            }
        ]
    }
    on_mid = SessionState(model="mid", quality=0.6, last_seen=0)
    r = route(config, "expert", body=body, session=on_mid)
    assert r.model.name == "mid" and "tool loop" in r.reason


def test_session_store_idle_and_lru(config: Config) -> None:
    store = SessionStore(max_entries=2)
    big = config.model("big")
    record(store, "a", big, now=100)
    assert store.get("a", now=150, idle_s=60) is not None
    assert store.get("a", now=1000, idle_s=60) is None
    record(store, "a", big, now=100)
    record(store, "b", big, now=100)
    record(store, "c", big, now=100, refused=True)
    assert len(store) == 2 and store.get("a", now=100, idle_s=60) is None
    state = store.get("c", now=100, idle_s=60)
    assert state is not None and state.escalate_next
    record(store, "d", big, sticky_mode="off")
    assert store.get("d", now=100, idle_s=60) is None


def test_route_as_dict(config: Config) -> None:
    out = route(config, "medium").as_dict()
    assert out["model"] == "mid" and out["backend"] == "lan"
    assert out["fallback_chain"] == ["mid", "big", "tiny"]
    assert out["decision"]["bucket"] == "medium"
