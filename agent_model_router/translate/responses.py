"""Non-streaming response translation and error envelopes for both APIs."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

__all__ = [
    "ANTHROPIC_STOP_TO_OPENAI",
    "OPENAI_FINISH_TO_ANTHROPIC",
    "Usage",
    "anthropic_error",
    "anthropic_to_openai_response",
    "error_for",
    "openai_error",
    "openai_to_anthropic_response",
    "usage_from",
]

OPENAI_FINISH_TO_ANTHROPIC = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}
ANTHROPIC_STOP_TO_OPENAI = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class Usage:
    __slots__ = ("input_tokens", "output_tokens", "known")

    def __init__(self, input_tokens: int = 0, output_tokens: int = 0, known: bool = False):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.known = known


def usage_from(api: str, body: Any) -> Usage:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return Usage()
    if api == "anthropic":
        inp = int(usage.get("input_tokens") or 0)
        inp += int(usage.get("cache_read_input_tokens") or 0)
        inp += int(usage.get("cache_creation_input_tokens") or 0)
        return Usage(inp, int(usage.get("output_tokens") or 0), True)
    return Usage(
        int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), True
    )


def _parse_args(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"_raw_arguments": str(raw)}
    return value if isinstance(value, dict) else {"value": value}


def openai_to_anthropic_response(body: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or "toolu_" + uuid.uuid4().hex[:24]),
                "name": str(fn.get("name")),
                "input": _parse_args(fn.get("arguments")),
            }
        )
    finish = choice.get("finish_reason") or "stop"
    stop = OPENAI_FINISH_TO_ANTHROPIC.get(finish, "end_turn")
    if message.get("tool_calls") and stop == "end_turn":
        stop = "tool_use"
    usage = usage_from("openai", body)
    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
    }


def anthropic_to_openai_response(body: dict[str, Any], model: str) -> dict[str, Any]:
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    for block in body.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(str(block.get("text", "")))
        elif block.get("type") == "tool_use":
            calls.append(
                {
                    "id": str(block.get("id")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name")),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
    if calls:
        message["tool_calls"] = calls
    usage = usage_from("anthropic", body)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": ANTHROPIC_STOP_TO_OPENAI.get(str(body.get("stop_reason")), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        },
    }


_ANTHROPIC_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def anthropic_error(status: int, message: str) -> dict[str, Any]:
    kind = _ANTHROPIC_TYPES.get(status, "api_error")
    return {"type": "error", "error": {"type": kind, "message": message}}


def openai_error(status: int, message: str) -> dict[str, Any]:
    kind = "invalid_request_error" if 400 <= status < 500 else "server_error"
    return {"error": {"message": message, "type": kind, "code": status}}


def error_for(api: str, status: int, message: str) -> dict[str, Any]:
    return anthropic_error(status, message) if api == "anthropic" else openai_error(status, message)


def upstream_error_message(body: bytes) -> str:
    """A short, safe message from an upstream error body (never the raw body)."""
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return "upstream error"
    err = data.get("error") if isinstance(data, dict) else None
    msg = err.get("message") if isinstance(err, dict) else err if isinstance(err, str) else None
    if not isinstance(msg, str):
        return "upstream error"
    return msg[:300]
