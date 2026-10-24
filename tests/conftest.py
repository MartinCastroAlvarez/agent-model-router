"""Shared fixtures: a three-backend pool and fake backends behind an httpx MockTransport.

No test opens a socket or calls a real provider: the router's httpx clients get a
``MockTransport`` whose handler plays ollama (OpenAI-compatible) and Anthropic.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import httpx
import pytest

from agent_model_router.config import Config
from agent_model_router.config import parse_config
from agent_model_router.metrics import MetricsStore
from agent_model_router.proxy import create_app

TOKEN = "amr-test-token-0123456789abcdef"
FAKE_KEY = "sk-ant-fake-key-for-tests-only"


def config_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "backends": {
            "local": {"kind": "ollama", "base_url": "http://127.0.0.1:11434/v1"},
            "lan": {
                "kind": "openai",
                "base_url": "http://gpu-box.lan:8000/v1",
                "allow_insecure": True,
            },
            "cloud": {
                "kind": "anthropic",
                "base_url": "https://api.anthropic.example",
                "api_key_env": "AMR_TEST_ANTHROPIC_KEY",
            },
        },
        "models": [
            {
                "name": "tiny",
                "backend": "local",
                "model": "qwen3:1.7b",
                "quality": 0.3,
                "price": {"input": 0.01, "output": 0.02},
                "context": 40000,
                "effort": "reasoning_effort",
            },
            {
                "name": "mid",
                "backend": "lan",
                "model": "qwen3:32b",
                "quality": 0.6,
                "price": {"input": 0.2, "output": 0.4},
                "context": 128000,
            },
            {
                "name": "big",
                "backend": "cloud",
                "model": "claude-sonnet-5-5",
                "quality": 0.9,
                "price": {"input": 2.0, "output": 10.0},
                "context": 1000000,
                "max_output": 64000,
                "effort": "adaptive",
            },
        ],
        "policy": {
            "quality": {"trivial": 0.0, "easy": 0.3, "medium": 0.55, "hard": 0.75, "expert": 0.85}
        },
        "metrics": {"enabled": False},
    }
    for key, value in overrides.items():
        data[key] = value
    return data


@pytest.fixture
def anthropic_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("AMR_TEST_ANTHROPIC_KEY", FAKE_KEY)
    return FAKE_KEY


@pytest.fixture
def config(anthropic_key: str) -> Config:
    return parse_config(config_dict())


async def _chunks(data: bytes, size: int = 7) -> AsyncIterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


def sse(events: list[tuple[str | None, Any]]) -> bytes:
    out = b""
    for name, data in events:
        payload = data if isinstance(data, str) else json.dumps(data)
        out += (f"event: {name}\n" if name else "").encode() + f"data: {payload}\n\n".encode()
    return out


def openai_stream_bytes(text: str = "Hello world", tool: bool = False) -> bytes:
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    events: list[tuple[str | None, Any]] = [
        (None, {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
    ]
    for word in text.split(" "):
        events.append((None, {**base, "choices": [{"index": 0, "delta": {"content": word + " "}}]}))
    finish = "stop"
    if tool:
        events.append(
            (
                None,
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "read_file", "arguments": ""},
                                    }
                                ]
                            },
                        }
                    ],
                },
            )
        )
        for frag in ('{"pa', 'th": "a', '.py"}'):
            events.append(
                (
                    None,
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{"index": 0, "function": {"arguments": frag}}]
                                },
                            }
                        ],
                    },
                )
            )
        finish = "tool_calls"
    events.append((None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}))
    events.append(
        (None, {**base, "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 5}})
    )
    events.append((None, "[DONE]"))
    return sse(events)


def anthropic_stream_bytes(
    text: str = "Hello world", stop: str = "end_turn", tool: bool = False
) -> bytes:
    events: list[tuple[str | None, Any]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 20, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    for word in text.split(" "):
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": word + " "},
                },
            )
        )
    events.append(("content_block_stop", {"type": "content_block_stop", "index": 1}))
    if tool:
        events.append(
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {},
                    },
                },
            )
        )
        for frag in ('{"path"', ': "b.py"}'):
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 2,
                        "delta": {"type": "input_json_delta", "partial_json": frag},
                    },
                )
            )
        events.append(("content_block_stop", {"type": "content_block_stop", "index": 2}))
        stop = "tool_use"
    events.append(("ping", {"type": "ping"}))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 9},
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return sse(events)


@dataclass
class FakeBackends:
    """Records every upstream request; per-host behaviour is configurable."""

    requests: list[httpx.Request] = field(default_factory=list)
    status: dict[str, int] = field(default_factory=dict)
    fail_connect: set[str] = field(default_factory=set)
    refuse: set[str] = field(default_factory=set)
    tool: bool = False
    text: str = "Hello world"

    def bodies(self, host: str | None = None) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if host is None or r.url.host == host]

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        if host in self.fail_connect:
            raise httpx.ConnectError("connection refused", request=request)
        if host in self.status:
            code = self.status[host]
            return httpx.Response(
                code,
                json={"error": {"message": f"fake {code}", "type": "x"}},
                headers={"retry-after": "1", "set-cookie": "leak=1"},
            )
        body = json.loads(request.content)
        stream = bool(body.get("stream"))
        if request.url.path.endswith("/v1/messages"):
            if stream:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream", "request-id": "req_1"},
                    content=_chunks(
                        anthropic_stream_bytes(
                            self.text,
                            "refusal" if host in self.refuse else "end_turn",
                            tool=self.tool,
                        )
                    ),
                )
            if host in self.refuse:
                return httpx.Response(
                    200,
                    json={
                        "id": "msg_r",
                        "type": "message",
                        "role": "assistant",
                        "model": body["model"],
                        "content": [],
                        "stop_reason": "refusal",
                        "usage": {"input_tokens": 3, "output_tokens": 0},
                    },
                )
            content: list[dict[str, Any]] = [{"type": "text", "text": self.text}]
            if self.tool:
                content.append(
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "b.py"},
                    }
                )
            return httpx.Response(
                200,
                headers={"request-id": "req_2"},
                json={
                    "id": "msg_2",
                    "type": "message",
                    "role": "assistant",
                    "model": body["model"],
                    "content": content,
                    "stop_reason": "tool_use" if self.tool else "end_turn",
                    "usage": {"input_tokens": 20, "output_tokens": 9, "cache_read_input_tokens": 5},
                },
            )
        if request.url.path.endswith("/chat/completions"):
            if stream:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_chunks(openai_stream_bytes(self.text, self.tool)),
                )
            text = "I can't help with that." if host in self.refuse else self.text
            message: dict[str, Any] = {"role": "assistant", "content": text}
            finish = "stop"
            if self.tool:
                message["tool_calls"] = [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    }
                ]
                finish = "tool_calls"
            return httpx.Response(
                200,
                json={
                    "id": "c2",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 5},
                },
            )
        return httpx.Response(404, json={"error": {"message": "nope"}})


@pytest.fixture
def fake() -> FakeBackends:
    return FakeBackends()


@dataclass
class Harness:
    client: httpx.AsyncClient
    fake: FakeBackends
    metrics: MetricsStore
    config: Config
    app: Any


AppFactory = Callable[..., Any]


@pytest.fixture
async def make_harness(anthropic_key: str, fake: FakeBackends) -> AsyncIterator[AppFactory]:
    opened: list[httpx.AsyncClient] = []

    async def factory(
        cfg: Config | None = None,
        *,
        token: str | None = TOKEN,
        clock: Callable[[], float] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Harness:
        cfg = cfg or parse_config(config_dict())
        metrics = MetricsStore(None)
        kwargs: dict[str, Any] = {}
        if clock is not None:
            kwargs["clock"] = clock
        app = create_app(
            cfg, token=token, transport=httpx.MockTransport(fake.handler), metrics=metrics, **kwargs
        )
        default_headers = {"x-api-key": TOKEN} if token else {}
        default_headers.update(headers or {})
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8787",
            headers=default_headers,
        )
        opened.append(client)
        return Harness(client=client, fake=fake, metrics=metrics, config=cfg, app=app)

    yield factory
    for client in opened:
        await client.aclose()


@pytest.fixture
async def h(make_harness: AppFactory) -> Harness:
    harness: Harness = await make_harness()
    return harness


def anthropic_body(text: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "claude-sonnet-5-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": text}],
    }
    body.update(extra)
    return body


def openai_body(text: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "auto", "messages": [{"role": "user", "content": text}]}
    body.update(extra)
    return body


TRIVIAL = "Rename get_usr to get_user in utils/users.py"
HARD = (
    "Refactor the plugin system to use dependency injection across the entire codebase, "
    "make it thread-safe and remove the race condition in the async scheduler."
)
EXPERT = (
    "Design a distributed, multi-tenant rate limiter with consensus and replication "
    "across regions; cover the architecture trade-offs, cache invalidation and a "
    "migration plan from the monolith."
)
