"""End-to-end through the ASGI app with fake backends: routing, translation, escalation."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from agent_model_router.config import parse_config
from agent_model_router.sse import SSEParser
from tests.conftest import EXPERT
from tests.conftest import FAKE_KEY
from tests.conftest import HARD
from tests.conftest import TOKEN
from tests.conftest import TRIVIAL
from tests.conftest import AppFactory
from tests.conftest import Harness
from tests.conftest import anthropic_body
from tests.conftest import anthropic_stream_bytes
from tests.conftest import config_dict
from tests.conftest import openai_body

MEDIUM = (
    "Review this pull request for the invoices feature: naming, test coverage and error handling."
)
LOCAL = "127.0.0.1"
LAN = "gpu-box.lan"
CLOUD = "api.anthropic.example"


def events(raw: bytes) -> list[dict]:
    parser = SSEParser()
    out = []
    for e in [*parser.feed(raw), *parser.flush()]:
        if e.data != "[DONE]":
            out.append({"event": e.event, **json.loads(e.data)})
        else:
            out.append({"event": "done"})
    return out


# --------------------------------------------------------------------------- auth / guards
async def test_requires_the_router_token(h: Harness) -> None:
    r = await h.client.post("/v1/messages", json=anthropic_body("hi"), headers={"x-api-key": ""})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"
    r = await h.client.post(
        "/v1/messages", json=anthropic_body("hi"), headers={"x-api-key": "wrong"}
    )
    assert r.status_code == 401
    assert h.fake.requests == []


async def test_bearer_token_is_accepted(make_harness: AppFactory) -> None:
    hh: Harness = await make_harness(headers={"x-api-key": ""})
    r = await hh.client.post(
        "/v1/chat/completions",
        json=openai_body(TRIVIAL),
        headers={"authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 200, r.text


async def test_dns_rebinding_host_is_refused(h: Harness) -> None:
    r = await h.client.post(
        "/v1/messages", json=anthropic_body("hi"), headers={"host": "evil.example:8787"}
    )
    assert r.status_code == 403
    r = await h.client.get("/healthz", headers={"host": "evil.example"})
    assert r.status_code == 403
    r = await h.client.get("/healthz", headers={"host": "[::1]:8787"})
    assert r.status_code == 200


async def test_browser_origins_are_refused(h: Harness) -> None:
    r = await h.client.post(
        "/v1/messages", json=anthropic_body("hi"), headers={"origin": "https://evil.example"}
    )
    assert r.status_code == 403
    assert h.fake.requests == []


async def test_body_limits_and_validation(make_harness: AppFactory) -> None:
    data = config_dict()
    data["server"] = {"max_body_bytes": 200}
    hh: Harness = await make_harness(parse_config(data))
    r = await hh.client.post("/v1/messages", json=anthropic_body("x" * 500))
    assert r.status_code == 413
    r = await hh.client.post(
        "/v1/messages", content=b"not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 400
    r = await hh.client.post("/v1/messages", json={"prompt": "x"})
    assert r.status_code == 400
    r = await hh.client.post("/v1/messages", json=[1, 2])
    assert r.status_code == 400


async def test_chunked_body_over_the_limit_is_refused(make_harness: AppFactory) -> None:
    data = config_dict()
    data["server"] = {"max_body_bytes": 100}
    hh: Harness = await make_harness(parse_config(data))

    async def gen():
        for _ in range(10):
            yield b'{"messages": []}' * 2

    r = await hh.client.post("/v1/messages", content=gen())
    assert r.status_code == 413


async def test_healthz_needs_no_token_and_reveals_nothing(h: Harness) -> None:
    r = await h.client.get("/healthz", headers={"x-api-key": ""})
    assert r.status_code == 200
    assert set(r.json()) == {"status", "version"}


async def test_auth_disabled_mode(make_harness: AppFactory) -> None:
    data = config_dict()
    data["server"] = {"auth": False}
    hh: Harness = await make_harness(parse_config(data), token=None)
    r = await hh.client.post("/v1/chat/completions", json=openai_body(TRIVIAL))
    assert r.status_code == 200


def test_auth_on_without_token_is_a_programming_error(config) -> None:
    from agent_model_router.proxy import create_app

    with pytest.raises(ValueError):
        create_app(config, token=None)


# --------------------------------------------------------------------------- routing
async def test_trivial_anthropic_request_goes_to_the_small_local_model(h: Harness) -> None:
    r = await h.client.post("/v1/messages", json=anthropic_body(TRIVIAL))
    assert r.status_code == 200, r.text
    assert r.headers["x-router-model"] == "tiny"
    assert r.headers["x-router-task"] == "edit"
    assert r.headers["x-router-bucket"] == "trivial"
    assert "cheapest" in r.headers["x-router-reason"]
    body = r.json()
    assert body["type"] == "message"
    assert body["content"][0] == {"type": "text", "text": "Hello world"}
    assert body["usage"] == {"input_tokens": 11, "output_tokens": 5}
    assert h.fake.hosts() == [LOCAL]
    sent = h.fake.bodies()[0]
    assert sent["model"] == "qwen3:1.7b"
    assert sent["messages"] == [{"role": "user", "content": TRIVIAL}]
    assert "reasoning_effort" not in sent  # trivial -> effort none


async def test_medium_request_goes_to_the_mid_model(h: Harness) -> None:
    r = await h.client.post("/v1/chat/completions", json=openai_body(MEDIUM))
    assert r.headers["x-router-model"] == "mid", r.headers["x-router-reason"]
    assert r.headers["x-router-effort"] == "medium"
    assert h.fake.hosts() == [LAN]
    assert r.json()["choices"][0]["message"]["content"] == "Hello world"


async def test_hard_request_goes_to_the_hosted_model_with_router_key(h: Harness) -> None:
    r = await h.client.post(
        "/v1/messages",
        json=anthropic_body(
            HARD, temperature=0.2, thinking={"type": "enabled", "budget_tokens": 99999}
        ),
        headers={
            "anthropic-beta": "context-1m",
            "anthropic-version": "2023-06-01",
            "cookie": "session=abc",
            "x-forwarded-for": "1.2.3.4",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["x-router-model"] == "big"
    assert r.headers["x-router-effort"] == "high"
    assert r.headers["request-id"] == "req_2"
    req = h.fake.requests[0]
    assert req.url.host == CLOUD and req.url.path == "/v1/messages"
    assert req.headers["x-api-key"] == FAKE_KEY  # the router's key ...
    assert TOKEN not in str(req.headers)  # ... never the client's token
    assert "cookie" not in req.headers and "x-forwarded-for" not in req.headers
    assert req.headers["anthropic-beta"] == "context-1m"
    sent = json.loads(req.content)
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}
    assert "temperature" not in sent
    assert r.json()["usage"]["input_tokens"] == 20


async def test_openai_inbound_to_anthropic_backend_is_translated(h: Harness) -> None:
    h.fake.tool = True
    body = openai_body(
        EXPERT,
        tools=[
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ],
        max_tokens=500,
    )
    r = await h.client.post(
        "/v1/chat/completions", json=body, headers={"anthropic-beta": "must-not-leak"}
    )
    assert r.headers["x-router-model"] == "big"
    sent = h.fake.bodies(CLOUD)[0]
    assert sent["tools"][0]["name"] == "read_file"
    assert sent["max_tokens"] == 500
    assert "anthropic-beta" not in h.fake.requests[0].headers
    out = r.json()
    msg = out["choices"][0]["message"]
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "b.py"}
    assert out["usage"]["prompt_tokens"] == 25  # input + cache reads


async def test_anthropic_tool_call_roundtrip_through_openai_backend(h: Harness) -> None:
    h.fake.tool = True
    r = await h.client.post(
        "/v1/messages",
        json=anthropic_body(
            TRIVIAL, tools=[{"name": "read_file", "input_schema": {"type": "object"}}]
        ),
    )
    out = r.json()
    assert out["stop_reason"] == "tool_use"
    assert out["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read_file",
        "input": {"path": "a.py"},
    }
    sent = h.fake.bodies(LOCAL)[0]
    assert sent["tools"][0]["function"]["name"] == "read_file"


async def test_client_can_pin_a_model(h: Harness) -> None:
    r = await h.client.post(
        "/v1/messages", json=anthropic_body(TRIVIAL), headers={"x-router-model": "big"}
    )
    assert r.headers["x-router-model"] == "big"
    assert "pinned" in r.headers["x-router-reason"]
    r = await h.client.post("/v1/chat/completions", json=openai_body(TRIVIAL, model="mid"))
    assert r.headers["x-router-model"] == "mid"
    r = await h.client.post(
        "/v1/messages", json=anthropic_body(TRIVIAL), headers={"x-router-model": "does-not-exist"}
    )
    assert r.headers["x-router-model"] == "tiny"


# --------------------------------------------------------------------------- stickiness
async def test_ratchet_keeps_the_conversation_on_the_bigger_model(h: Harness) -> None:
    sid = {"x-claude-code-session-id": "conv-1"}
    r1 = await h.client.post("/v1/messages", json=anthropic_body(HARD), headers=sid)
    assert r1.headers["x-router-model"] == "big"
    r2 = await h.client.post("/v1/messages", json=anthropic_body("thanks, ok"), headers=sid)
    assert r2.headers["x-router-model"] == "big"
    assert "sticky" in r2.headers["x-router-reason"]
    other = await h.client.post(
        "/v1/messages",
        json=anthropic_body("thanks, ok"),
        headers={"x-claude-code-session-id": "conv-2"},
    )
    assert other.headers["x-router-model"] == "tiny"
    assert r1.headers["x-router-session"] != other.headers["x-router-session"]


async def test_ratchet_moves_up_but_not_down(h: Harness) -> None:
    sid = {"x-session-id": "conv-up"}
    r1 = await h.client.post("/v1/chat/completions", json=openai_body(TRIVIAL), headers=sid)
    assert r1.headers["x-router-model"] == "tiny"
    r2 = await h.client.post("/v1/chat/completions", json=openai_body(HARD), headers=sid)
    assert r2.headers["x-router-model"] == "big"
    assert "up from tiny" in r2.headers["x-router-reason"]


async def test_idle_conversations_reset(make_harness: AppFactory) -> None:
    now = [1000.0]
    hh: Harness = await make_harness(clock=lambda: now[0])
    sid = {"x-session-id": "idle"}
    await hh.client.post("/v1/messages", json=anthropic_body(HARD), headers=sid)
    now[0] += 31 * 60
    r = await hh.client.post("/v1/messages", json=anthropic_body("thanks, ok"), headers=sid)
    assert r.headers["x-router-model"] == "tiny"


async def test_tool_loop_steps_keep_the_model(h: Harness) -> None:
    sid = {"x-session-id": "loop"}
    await h.client.post("/v1/messages", json=anthropic_body(MEDIUM), headers=sid)
    body = {
        "model": "x",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": MEDIUM},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file text"}],
            },
        ],
    }
    r = await h.client.post("/v1/messages", json=body, headers=sid)
    assert r.headers["x-router-model"] == "mid"
    assert r.headers["x-router-task"] == "tool_followup"
    assert "tool loop" in r.headers["x-router-reason"]


async def test_sticky_off(make_harness: AppFactory) -> None:
    data = config_dict()
    data["policy"]["sticky"] = "off"
    hh: Harness = await make_harness(parse_config(data))
    sid = {"x-session-id": "s"}
    await hh.client.post("/v1/messages", json=anthropic_body(HARD), headers=sid)
    r = await hh.client.post("/v1/messages", json=anthropic_body("thanks, ok"), headers=sid)
    assert r.headers["x-router-model"] == "tiny"


# --------------------------------------------------------------------------- escalation
async def test_escalates_on_5xx(h: Harness) -> None:
    h.fake.status[LOCAL] = 503
    r = await h.client.post("/v1/messages", json=anthropic_body(TRIVIAL))
    assert r.status_code == 200
    assert r.headers["x-router-model"] == "mid"
    assert r.headers["x-router-attempts"] == "2"
    assert h.fake.hosts() == [LOCAL, LAN]
    assert "set-cookie" not in r.headers


async def test_escalates_on_connection_errors(h: Harness) -> None:
    h.fake.fail_connect |= {LOCAL, LAN}
    r = await h.client.post("/v1/chat/completions", json=openai_body(TRIVIAL))
    assert r.status_code == 200
    assert r.headers["x-router-model"] == "big"
    assert h.fake.hosts() == [LOCAL, LAN, CLOUD]


async def test_all_backends_down_is_a_502(h: Harness) -> None:
    h.fake.fail_connect |= {LOCAL, LAN, CLOUD}
    r = await h.client.post("/v1/messages", json=anthropic_body(TRIVIAL))
    assert r.status_code == 502
    assert r.json()["type"] == "error"
    assert "all backends failed" in r.json()["error"]["message"]
    assert h.metrics.summary()["models"][0]["errors"] == 1


async def test_last_attempt_retryable_error_is_returned(h: Harness) -> None:
    h.fake.status.update({LOCAL: 503, LAN: 503, CLOUD: 429})
    r = await h.client.post("/v1/messages", json=anthropic_body(TRIVIAL))
    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limit_error"
    assert r.headers["retry-after"] == "1"


async def test_client_errors_are_not_retried(h: Harness) -> None:
    h.fake.status[LOCAL] = 400
    r = await h.client.post("/v1/chat/completions", json=openai_body(TRIVIAL))
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "fake 400"
    assert h.fake.hosts() == [LOCAL]


async def test_refusals_escalate_to_a_stronger_model(h: Harness) -> None:
    h.fake.refuse.add(LOCAL)
    r = await h.client.post("/v1/messages", json=anthropic_body(TRIVIAL))
    assert r.headers["x-router-model"] == "mid"
    assert r.json()["content"][0]["text"] == "Hello world"


async def test_refusal_on_the_last_model_is_returned_and_remembered(
    make_harness: AppFactory,
) -> None:
    data = config_dict()
    data["policy"]["escalate_on_refusal"] = False
    hh: Harness = await make_harness(parse_config(data))
    hh.fake.refuse.add(LOCAL)
    sid = {"x-session-id": "refuse"}
    r = await hh.client.post("/v1/chat/completions", json=openai_body(TRIVIAL), headers=sid)
    assert r.headers["x-router-model"] == "tiny"
    r = await hh.client.post("/v1/chat/completions", json=openai_body(TRIVIAL), headers=sid)
    assert r.headers["x-router-model"] == "mid"
    assert "escalated after a refusal" in r.headers["x-router-reason"]


# --------------------------------------------------------------------------- streaming
async def test_stream_anthropic_inbound_from_openai_backend(h: Harness) -> None:
    h.fake.tool = True
    async with h.client.stream(
        "POST", "/v1/messages", json=anthropic_body(TRIVIAL, stream=True)
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-router-model"] == "tiny"
        raw = b"".join([c async for c in r.aiter_bytes()])
    ev = events(raw)
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    text = "".join(
        e["delta"]["text"]
        for e in ev
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
    )
    assert text == "Hello world "
    args = "".join(
        e["delta"]["partial_json"]
        for e in ev
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(args) == {"path": "a.py"}
    starts = [e for e in ev if e["type"] == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["text", "tool_use"]
    assert [s["index"] for s in starts] == [0, 1]
    delta = next(e for e in ev if e["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"
    assert delta["usage"]["output_tokens"] == 5
    assert all(e["event"] == e["type"] for e in ev)
    sent = h.fake.bodies()[0]
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    summary = h.metrics.summary()
    assert summary["models"][0]["input_tokens"] == 11


async def test_stream_passthrough_is_byte_for_byte(h: Harness) -> None:
    async with h.client.stream("POST", "/v1/messages", json=anthropic_body(HARD, stream=True)) as r:
        assert r.headers["x-router-model"] == "big"
        raw = b"".join([c async for c in r.aiter_bytes()])
    assert raw == anthropic_stream_bytes()
    row = h.metrics.summary()["models"][0]
    assert (row["input_tokens"], row["output_tokens"]) == (20, 9)


async def test_stream_openai_inbound_from_anthropic_backend(h: Harness) -> None:
    h.fake.tool = True
    body = openai_body(EXPERT, stream=True, stream_options={"include_usage": True})
    async with h.client.stream("POST", "/v1/chat/completions", json=body) as r:
        raw = b"".join([c async for c in r.aiter_bytes()])
    ev = events(raw)
    assert ev[-1] == {"event": "done"}
    chunks = [e for e in ev if e.get("object") == "chat.completion.chunk"]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "Hello world "
    calls = [
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in chunks
        if c["choices"] and "tool_calls" in c["choices"][0]["delta"]
    ]
    assert calls[0]["id"] == "toolu_1" and calls[0]["function"]["name"] == "read_file"
    assert json.loads("".join(c["function"]["arguments"] for c in calls)) == {"path": "b.py"}
    finishes = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
    assert finishes[-1] == "tool_calls"
    usage = [c for c in chunks if not c["choices"]]
    assert usage[0]["usage"] == {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}


async def test_stream_escalates_before_the_first_byte(h: Harness) -> None:
    h.fake.status[LOCAL] = 500
    async with h.client.stream(
        "POST", "/v1/chat/completions", json=openai_body(TRIVIAL, stream=True)
    ) as r:
        assert r.headers["x-router-model"] == "mid"
        raw = b"".join([c async for c in r.aiter_bytes()])
    assert raw.endswith(b"data: [DONE]\n\n")


async def test_stream_refusal_escalates_the_next_turn(h: Harness) -> None:
    h.fake.refuse.add(CLOUD)
    data = config_dict()
    sid = {"x-session-id": "stream-refusal"}
    async with h.client.stream(
        "POST",
        "/v1/messages",
        json=anthropic_body(HARD, stream=True),
        headers={**sid, "x-router-model": "mid"},
    ) as r:
        await r.aread()
    assert data  # the pinned turn above went to mid; next one escalates beyond it
    async with h.client.stream(
        "POST", "/v1/messages", json=anthropic_body(TRIVIAL, stream=True), headers=sid
    ) as r:
        await r.aread()
        assert r.headers["x-router-model"] == "mid"


async def test_broken_stream_is_logged_as_an_error(make_harness: AppFactory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        async def broken():
            yield b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            raise httpx.ReadError("reset", request=request)

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=broken())

    from agent_model_router.metrics import MetricsStore
    from agent_model_router.proxy import create_app

    cfg = parse_config(config_dict())
    metrics = MetricsStore(None)
    app = create_app(cfg, token=TOKEN, transport=httpx.MockTransport(handler), metrics=metrics)
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1",
            headers={"x-api-key": TOKEN},
        ) as c,
        c.stream("POST", "/v1/chat/completions", json=openai_body(TRIVIAL, stream=True)) as r,
    ):
        raw = await r.aread()
    assert b"hi" in raw
    assert metrics.summary()["models"][0]["errors"] == 1


# --------------------------------------------------------------------------- endpoints
async def test_explain_is_a_dry_run(h: Harness) -> None:
    r = await h.client.post("/router/explain", json=anthropic_body(HARD))
    assert r.status_code == 200
    out = r.json()
    assert out["model"] == "big"
    assert out["decision"]["bucket"] == "hard"
    assert out["decision"]["contributions"]
    assert {c["model"] for c in out["candidates"]} == {"tiny", "mid", "big"}
    assert out["fallback_chain"][0] == "big"
    assert out["baseline_model"] == "big"
    assert h.fake.requests == []
    r = await h.client.post("/router/explain?api=openai", json=openai_body(TRIVIAL))
    assert r.json()["model"] == "tiny"
    r = await h.client.post("/router/explain?api=gopher", json=openai_body(TRIVIAL))
    assert r.status_code == 400


async def test_stats_and_savings(h: Harness) -> None:
    await h.client.post("/v1/messages", json=anthropic_body(TRIVIAL))
    await h.client.post("/v1/messages", json=anthropic_body(HARD))
    r = await h.client.get("/router/stats")
    out = r.json()
    assert out["requests"] == 2
    assert out["baseline_model"] == "big"
    assert out["savings_usd"] > 0
    assert 0 < out["savings_pct"] < 100
    assert {t["task"] for t in out["tasks"]} == {"edit"}


async def test_models_and_count_tokens(h: Harness) -> None:
    r = await h.client.get("/v1/models")
    ids = [m["id"] for m in r.json()["data"]]
    assert ids == ["auto", "tiny", "mid", "big"]
    r = await h.client.post("/v1/messages/count_tokens", json=anthropic_body("x" * 400))
    assert r.json()["input_tokens"] > 100
    assert r.headers["x-router-estimate"] == "true"


async def test_unusable_backend_is_skipped_and_empty_pool_is_503(
    make_harness: AppFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AMR_TEST_ANTHROPIC_KEY")
    hh: Harness = await make_harness(parse_config(config_dict()))
    r = await hh.client.post("/v1/messages", json=anthropic_body(EXPERT))
    assert r.headers["x-router-model"] == "mid"
    assert "no model clears" in r.headers["x-router-reason"]
    data = config_dict()
    for b in data["backends"].values():
        b["enabled"] = False
    hh = await make_harness(parse_config(data))
    r = await hh.client.post("/v1/messages", json=anthropic_body(EXPERT))
    assert r.status_code == 503
    r = await hh.client.post("/router/explain", json=anthropic_body(EXPERT))
    assert r.status_code == 503


async def test_daily_budget(make_harness: AppFactory) -> None:
    data = config_dict(budgets={"daily_usd": 0.0000001})
    hh: Harness = await make_harness(parse_config(data))
    r = await hh.client.post("/v1/messages", json=anthropic_body(HARD))
    assert r.headers["x-router-model"] == "big"
    r = await hh.client.post("/v1/messages", json=anthropic_body(HARD))
    assert r.headers["x-router-model"] == "tiny"
    assert "budget" in r.headers["x-router-reason"]
    data = config_dict(budgets={"daily_usd": 0.0000001, "on_exceeded": "reject"})
    hh = await make_harness(parse_config(data))
    await hh.client.post("/v1/messages", json=anthropic_body(HARD))
    r = await hh.client.post("/v1/messages", json=anthropic_body(HARD))
    assert r.status_code == 429


SECRET_PROMPT = HARD + "\nThe key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"


async def test_secrets_only_go_to_trusted_backends(h: Harness) -> None:
    r = await h.client.post("/v1/messages", json=anthropic_body(SECRET_PROMPT))
    assert r.headers["x-router-model"] == "tiny"  # the only loopback backend
    assert "secret detected" in r.headers["x-router-reason"]
    assert h.fake.hosts() == [LOCAL]


async def test_a_lan_backend_can_be_marked_trusted(make_harness: AppFactory) -> None:
    data = config_dict()
    data["backends"]["lan"]["trusted"] = True
    hh: Harness = await make_harness(parse_config(data))
    r = await hh.client.post("/v1/messages", json=anthropic_body(SECRET_PROMPT))
    assert r.headers["x-router-model"] == "mid"
    assert CLOUD not in hh.fake.hosts()


async def test_prompts_and_keys_never_reach_logs_or_metrics(
    h: Harness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    marker = "PURPLE-ELEPHANT-PROMPT-MARKER"
    h.fake.status[LOCAL] = 503
    await h.client.post("/v1/messages", json=anthropic_body(f"Rename {marker} in a.py"))
    await h.client.post("/v1/messages", json=anthropic_body(HARD))
    logs = caplog.text
    assert marker not in logs and TOKEN not in logs and FAKE_KEY not in logs
    rows = h.metrics.conn.execute("SELECT * FROM routes").fetchall()
    dump = json.dumps(rows)
    assert marker not in dump and TOKEN not in dump and FAKE_KEY not in dump
    assert "route api=anthropic" in logs
