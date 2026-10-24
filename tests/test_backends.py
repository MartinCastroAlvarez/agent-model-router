"""Effort application and upstream request building (respx for the HTTP layer)."""

from __future__ import annotations

import httpx
import pytest
import respx

from agent_model_router.backends import Upstream
from agent_model_router.backends import apply_effort
from agent_model_router.backends import prepare_body
from agent_model_router.config import Config
from agent_model_router.config import Model
from agent_model_router.config import parse_config
from tests.conftest import FAKE_KEY
from tests.conftest import config_dict


def model(style: str, max_output: int = 64000) -> Model:
    return Model(
        name="m", backend="b", model="x", quality=0.5, max_output=max_output, effort_style=style
    )


def test_effort_none_strips_client_reasoning(config: Config) -> None:
    body = {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "reasoning": {},
        "output_config": {"effort": "max", "format": {"type": "json"}},
    }
    out = apply_effort(body, "anthropic", model("none"), "high", config.policy)
    assert out == {"output_config": {"format": {"type": "json"}}}
    out = apply_effort(
        {"output_config": {"effort": "max"}}, "openai", model("none"), "high", config.policy
    )
    assert out == {}


def test_reasoning_effort_style(config: Config) -> None:
    assert apply_effort({}, "openai", model("reasoning_effort"), "low", config.policy) == {
        "reasoning_effort": "low"
    }
    assert apply_effort({}, "openai", model("reasoning_effort"), "none", config.policy) == {}
    # an anthropic-style setting on an openai backend still maps to reasoning_effort
    assert apply_effort({}, "openai", model("adaptive"), "high", config.policy) == {
        "reasoning_effort": "high"
    }


def test_adaptive_style(config: Config) -> None:
    out = apply_effort(
        {"temperature": 0.3, "top_k": 3}, "anthropic", model("adaptive"), "medium", config.policy
    )
    assert out == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    out = apply_effort({}, "anthropic", model("adaptive"), "none", config.policy)
    assert out["output_config"] == {"effort": "low"}  # current models cannot turn it off


def test_budget_style(config: Config) -> None:
    out = apply_effort(
        {"max_tokens": 1000, "temperature": 0.5},
        "anthropic",
        model("budget"),
        "medium",
        config.policy,
    )
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert out["max_tokens"] > 8192 and "temperature" not in out
    out = apply_effort({"max_tokens": 50000}, "anthropic", model("budget"), "low", config.policy)
    assert out["max_tokens"] == 50000 and out["thinking"]["budget_tokens"] == 2048
    capped = apply_effort(
        {"max_tokens": 1000}, "anthropic", model("budget", max_output=6000), "high", config.policy
    )
    assert capped["thinking"]["budget_tokens"] == 6000 - 1024 and capped["max_tokens"] == 6000
    tiny = apply_effort(
        {"max_tokens": 1000}, "anthropic", model("budget", max_output=1500), "high", config.policy
    )
    assert "thinking" not in tiny
    assert "thinking" not in apply_effort({}, "anthropic", model("budget"), "none", config.policy)
    forced = apply_effort(
        {"tool_choice": {"type": "tool", "name": "f"}},
        "anthropic",
        model("budget"),
        "high",
        config.policy,
    )
    assert "thinking" not in forced


def test_prepare_body_clamps_and_swaps_model(config: Config) -> None:
    big = config.model("big")
    body = {"model": "whatever", "max_tokens": 999999, "messages": []}
    out = prepare_body("anthropic", body, big, config.backends["cloud"], "low", config.policy)
    assert out["model"] == "claude-sonnet-5-5" and out["max_tokens"] == 64000
    assert body["model"] == "whatever"  # the client's body is not mutated
    tiny = config.model("tiny")
    out = prepare_body(
        "openai",
        {"messages": [], "max_completion_tokens": 999999},
        tiny,
        config.backends["local"],
        "none",
        config.policy,
    )
    assert out["max_completion_tokens"] == tiny.max_output


@respx.mock
async def test_upstream_sends_the_router_key_and_no_redirects(config: Config) -> None:
    route = respx.post("https://api.anthropic.example/v1/messages").mock(
        return_value=httpx.Response(302, headers={"location": "https://evil.example/"})
    )
    upstream = Upstream(config)
    backend = config.backends["cloud"]
    request = upstream.build(
        backend, {"model": "x", "stream": True}, {"x-api-key": "client"}, "anthropic"
    )
    response = await upstream.send(backend, request, stream=False)
    assert response.status_code == 302  # not followed
    sent = route.calls.last.request
    assert sent.headers["x-api-key"] == FAKE_KEY
    assert sent.headers["accept"] == "text/event-stream"
    await upstream.aclose()


@respx.mock
async def test_complete_text_for_both_dialects(anthropic_key: str) -> None:
    respx.post("http://127.0.0.1:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": '{"task": "edit"}'}}]}
        )
    )
    respx.post("https://api.anthropic.example/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "hello"}]
            },
        )
    )
    cfg = parse_config(config_dict())
    upstream = Upstream(cfg)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert await upstream.complete_text(cfg.model("tiny"), msgs) == '{"task": "edit"}'
    assert await upstream.complete_text(cfg.model("big"), msgs) == "hello"
    sent = respx.calls[1].request
    assert b'"system":"s"' in sent.content.replace(b" ", b"")
    await upstream.aclose()


@respx.mock
async def test_complete_text_raises_on_errors(config: Config) -> None:
    respx.post("http://127.0.0.1:11434/v1/chat/completions").mock(return_value=httpx.Response(500))
    upstream = Upstream(config)
    with pytest.raises(httpx.HTTPStatusError):
        await upstream.complete_text(config.model("tiny"), [{"role": "user", "content": "x"}])
    await upstream.aclose()


@respx.mock
async def test_decision_mode_end_to_end(make_harness) -> None:
    respx.post("http://127.0.0.1:11434/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"task": "plan", "difficulty": "expert"}'}}
                    ]
                },
            ),
        ]
    )
    data = config_dict(classifier={"mode": "decision", "decision_model": "tiny"})
    cfg = parse_config(data)
    from agent_model_router.proxy import create_app

    app = create_app(cfg, token=None if not cfg.server.auth else "t")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={"x-api-key": "t"},
    ) as c:
        r = await c.post(
            "/router/explain",
            json={"messages": [{"role": "user", "content": "Rename foo to bar in a.py"}]},
        )
    out = r.json()
    assert out["decision"]["source"] == "decision"
    assert out["decision"]["bucket"] == "expert" and out["model"] == "big"
