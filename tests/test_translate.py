"""Translation: requests, responses and SSE streams in both directions."""

from __future__ import annotations

import json
import random

from agent_model_router.sse import SSEParser
from agent_model_router.sse import encode_event
from agent_model_router.translate import anthropic_to_openai_request
from agent_model_router.translate import anthropic_to_openai_response
from agent_model_router.translate import make_adapter
from agent_model_router.translate import openai_to_anthropic_request
from agent_model_router.translate import openai_to_anthropic_response
from agent_model_router.translate.responses import error_for
from agent_model_router.translate.responses import upstream_error_message
from agent_model_router.translate.responses import usage_from
from agent_model_router.translate.streams import AnthropicToOpenAI
from agent_model_router.translate.streams import OpenAIToAnthropic
from agent_model_router.translate.streams import Passthrough
from tests.conftest import anthropic_stream_bytes
from tests.conftest import openai_stream_bytes
from tests.conftest import sse

ANTHROPIC_CONVERSATION = {
    "model": "claude",
    "max_tokens": 300,
    "system": [{"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}}],
    "temperature": 0.1,
    "stop_sequences": ["END"],
    "stream": True,
    "metadata": {"user_id": "u"},
    "tools": [
        {"name": "read_file", "description": "Read", "input_schema": {"type": "object"}},
        {"type": "web_search_20260209", "name": "web_search"},
    ],
    "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "print(1)"}],
                },
                {"type": "tool_result", "tool_use_id": "t2", "content": "boom", "is_error": True},
                {"type": "text", "text": "and now?"},
            ],
        },
    ],
}


def test_anthropic_request_to_openai() -> None:
    out = anthropic_to_openai_request(ANTHROPIC_CONVERSATION, "qwen3")
    assert out["model"] == "qwen3"
    assert out["max_tokens"] == 300 and out["temperature"] == 0.1 and out["stop"] == ["END"]
    assert out["stream"] is True and out["stream_options"] == {"include_usage": True}
    msgs = out["messages"]
    assert msgs[0] == {"role": "system", "content": "Be terse."}
    assert msgs[1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    assert msgs[2]["content"] == "Reading."
    assert msgs[2]["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": '{"path": "a.py"}',
    }
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "print(1)"}
    assert msgs[4] == {"role": "tool", "tool_call_id": "t2", "content": "[tool error] boom"}
    assert msgs[5] == {"role": "user", "content": "and now?"}
    assert [t["function"]["name"] for t in out["tools"]] == ["read_file"]  # server tool dropped
    assert out["tool_choice"] == "required" and out["parallel_tool_calls"] is False
    assert "metadata" not in out


def test_anthropic_tool_choice_variants() -> None:
    base = {"messages": [], "tools": [{"name": "f", "input_schema": {}}]}
    for choice, want in [
        ({"type": "auto"}, "auto"),
        ({"type": "none"}, "none"),
        ({"type": "tool", "name": "f"}, {"type": "function", "function": {"name": "f"}}),
    ]:
        assert (
            anthropic_to_openai_request({**base, "tool_choice": choice}, "m")["tool_choice"] == want
        )
    assert "tool_choice" not in anthropic_to_openai_request({**base, "tool_choice": "x"}, "m")


def test_anthropic_url_images_and_plain_strings() -> None:
    out = anthropic_to_openai_request(
        {
            "system": "S",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": "https://x/i.png"}}
                    ],
                },
                {"role": "assistant", "content": "plain"},
                {"role": "user", "content": [{"type": "image", "source": {"type": "file"}}]},
                "garbage",
            ],
        },
        "m",
    )
    assert out["messages"][1]["content"][0]["image_url"]["url"] == "https://x/i.png"
    assert out["messages"][2] == {"role": "assistant", "content": "plain"}
    assert len(out["messages"]) == 3


OPENAI_CONVERSATION = {
    "model": "auto",
    "max_completion_tokens": 222,
    "stop": "END",
    "top_p": 0.9,
    "stream": True,
    "parallel_tool_calls": False,
    "tool_choice": "required",
    "tools": [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
    ],
    "messages": [
        {"role": "system", "content": "Be terse."},
        {"role": "developer", "content": [{"type": "text", "text": "Use tools."}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBBB"}},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                {"type": "image_url", "image_url": {"url": "http://insecure/y.png"}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "not json"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "one"},
        {"role": "tool", "tool_call_id": "c2", "content": "two"},
        {"role": "user", "content": "next"},
    ],
}


def test_openai_request_to_anthropic() -> None:
    out = openai_to_anthropic_request(OPENAI_CONVERSATION, "claude")
    assert out["system"] == "Be terse.\n\nUse tools."
    assert out["max_tokens"] == 222 and out["stop_sequences"] == ["END"] and out["top_p"] == 0.9
    user = out["messages"][0]
    assert user["role"] == "user"
    assert user["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "BBBB"},
    }
    assert user["content"][2]["source"] == {"type": "url", "url": "https://x/y.png"}
    assert len(user["content"]) == 3  # plain-http image dropped
    asst = out["messages"][1]
    assert asst["content"][0] == {
        "type": "tool_use",
        "id": "c1",
        "name": "read_file",
        "input": {"path": "a"},
    }
    assert asst["content"][1]["input"] == {"_raw_arguments": "not json"}
    results = out["messages"][2]
    assert results["role"] == "user"
    assert [b["type"] for b in results["content"]] == ["tool_result", "tool_result", "text"]
    assert out["tools"][0] == {
        "name": "read_file",
        "description": "",
        "input_schema": {"type": "object"},
    }
    assert out["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    assert out["stream"] is True


def test_openai_request_defaults_and_choices() -> None:
    out = openai_to_anthropic_request({"messages": [{"role": "user", "content": "hi"}]}, "m")
    assert out["max_tokens"] == 4096 and "system" not in out
    tools = [{"type": "function", "function": {"name": "f"}}]
    for choice, want in [
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ({"type": "function", "function": {"name": "f"}}, {"type": "tool", "name": "f"}),
    ]:
        got = openai_to_anthropic_request(
            {"messages": [], "tools": tools, "tool_choice": choice}, "m"
        )
        assert got["tool_choice"] == want
    got = openai_to_anthropic_request({"messages": [], "tools": tools, "stop": ["a", "b"]}, "m")
    assert got["stop_sequences"] == ["a", "b"] and "tool_choice" not in got


def test_responses_both_ways() -> None:
    oa = {
        "choices": [
            {
                "message": {
                    "content": "hi",
                    "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "[1]"}}],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    an = openai_to_anthropic_response(oa, "m")
    assert an["stop_reason"] == "tool_use"
    assert an["content"][1]["input"] == {"value": [1]}
    assert an["usage"] == {"input_tokens": 3, "output_tokens": 4}
    an2 = openai_to_anthropic_response(
        {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "x"}]}, "finish_reason": "length"}
            ]
        },
        "m",
    )
    assert an2["stop_reason"] == "max_tokens" and an2["content"][0]["text"] == "x"
    back = anthropic_to_openai_response(
        {
            "content": [{"type": "text", "text": "a"}, "junk", {"type": "text", "text": "b"}],
            "stop_reason": "refusal",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
        "m",
    )
    assert back["choices"][0]["message"] == {"role": "assistant", "content": "ab"}
    assert back["choices"][0]["finish_reason"] == "content_filter"
    assert back["usage"]["total_tokens"] == 3


def test_usage_and_errors() -> None:
    assert not usage_from("openai", {}).known
    assert not usage_from("openai", "x").known
    assert (
        usage_from(
            "anthropic",
            {"usage": {"input_tokens": 1, "cache_creation_input_tokens": 2, "output_tokens": 3}},
        ).input_tokens
        == 3
    )
    assert error_for("anthropic", 529, "x")["error"]["type"] == "overloaded_error"
    assert error_for("anthropic", 418, "x")["error"]["type"] == "api_error"
    assert error_for("openai", 500, "x")["error"]["type"] == "server_error"
    assert upstream_error_message(b"not json") == "upstream error"
    assert upstream_error_message(b'{"error": "plain"}') == "plain"
    assert upstream_error_message(b'{"error": {"message": 5}}') == "upstream error"
    assert (
        len(upstream_error_message(json.dumps({"error": {"message": "x" * 999}}).encode())) == 300
    )


# --------------------------------------------------------------------------- SSE
def test_sse_parser_handles_splits_crlf_comments_and_flush() -> None:
    raw = b': keepalive\r\n\r\nevent: a\r\ndata: {"x":1}\r\n\r\ndata: line1\ndata: line2\n\ndata: tail'
    for size in (1, 2, 3, 5, 64):
        parser = SSEParser()
        got = []
        for i in range(0, len(raw), size):
            got += list(parser.feed(raw[i : i + size]))
        got += list(parser.flush())
        assert [(e.event, e.data) for e in got] == [
            ("a", '{"x":1}'),
            (None, "line1\nline2"),
            (None, "tail"),
        ]
    assert list(SSEParser().flush()) == []
    assert encode_event("[DONE]") == b"data: [DONE]\n\n"
    assert encode_event({"a": 1}, "x") == b'event: x\ndata: {"a":1}\n\n'


def _run(adapter, raw: bytes, seed: int = 0) -> bytes:
    rnd = random.Random(seed)
    out = b""
    i = 0
    while i < len(raw):
        n = rnd.randint(1, 40)
        out += adapter.feed(raw[i : i + n])
        i += n
    return out + adapter.finish()


def _events(raw: bytes) -> list[dict]:
    p = SSEParser()
    return [json.loads(e.data) for e in [*p.feed(raw), *p.flush()] if e.data != "[DONE]"]


def test_openai_to_anthropic_stream_at_random_split_points() -> None:
    raw = openai_stream_bytes("one two three", tool=True)
    for seed in range(5):
        adapter = OpenAIToAnthropic("qwen")
        ev = _events(_run(adapter, raw, seed))
        assert ev[0]["type"] == "message_start" and ev[0]["message"]["model"] == "qwen"
        assert ev[-1]["type"] == "message_stop"
        text = "".join(e["delta"].get("text", "") for e in ev if e["type"] == "content_block_delta")
        assert text == "one two three "
        stops = [e for e in ev if e["type"] == "content_block_stop"]
        assert [s["index"] for s in stops] == [0, 1]
        assert adapter.usage.input_tokens == 11 and adapter.stop_reason == "tool_use"


def test_openai_to_anthropic_stream_edge_cases() -> None:
    adapter = OpenAIToAnthropic("m")
    assert _events(adapter.finish())[-1]["type"] == "message_stop"  # empty upstream
    assert adapter.finish() == b""
    adapter = OpenAIToAnthropic("m")
    raw = sse(
        [
            (None, {"error": {"message": "overloaded"}}),
            (None, "not json"),
            (
                None,
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "a",
                                        "function": {"name": "f", "arguments": "{}"},
                                    }
                                ]
                            }
                        }
                    ]
                },
            ),
            (
                None,
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 1, "id": "b", "function": {"name": "g"}}]
                            }
                        }
                    ]
                },
            ),
            (
                None,
                {
                    "choices": [
                        {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "late"}}]}}
                    ]
                },
            ),
            (None, {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]
    )
    ev = _events(_run(adapter, raw))
    assert ev[1] == {"type": "error", "error": {"type": "api_error", "message": "overloaded"}}
    assert "late" not in json.dumps(ev)
    assert ev[-2]["delta"]["stop_reason"] == "tool_use"
    assert adapter.error == "error"


def test_anthropic_to_openai_stream_at_random_split_points() -> None:
    raw = anthropic_stream_bytes("a b", tool=True)
    for seed in range(5):
        adapter = AnthropicToOpenAI("claude", include_usage=True)
        out = _run(adapter, raw, seed)
        assert out.endswith(b"data: [DONE]\n\n") and out.count(b"[DONE]") == 1
        chunks = _events(out)
        assert chunks[-1]["usage"]["completion_tokens"] == 9
        assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_to_openai_stream_errors_and_truncation() -> None:
    adapter = AnthropicToOpenAI("m")
    raw = sse(
        [
            ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 9,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                },
            ),
        ]
    )
    out = _run(adapter, raw)
    assert _events(out)[0] == {"error": {"message": "busy", "type": "overloaded_error"}}
    assert out.endswith(b"data: [DONE]\n\n")  # closed even though message_stop never came
    assert adapter.error == "overloaded_error"


def test_passthrough_tracks_usage_without_changing_bytes() -> None:
    raw = anthropic_stream_bytes("x y", stop="refusal")
    adapter = Passthrough("anthropic")
    assert _run(adapter, raw) == raw
    assert (adapter.usage.input_tokens, adapter.usage.output_tokens) == (20, 9)
    assert adapter.stop_reason == "refusal"
    raw = openai_stream_bytes("p q")
    adapter = Passthrough("openai")
    assert _run(adapter, raw) == raw
    assert adapter.usage.output_tokens == 5 and adapter.stop_reason == "end_turn"
    assert isinstance(make_adapter("openai", "openai", "m"), Passthrough)
    assert isinstance(make_adapter("anthropic", "openai", "m"), OpenAIToAnthropic)
    assert isinstance(make_adapter("openai", "anthropic", "m"), AnthropicToOpenAI)
