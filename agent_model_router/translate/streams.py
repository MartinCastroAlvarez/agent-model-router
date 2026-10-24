"""Streaming (SSE) adapters.

Every adapter takes upstream bytes and returns the bytes to send downstream, while it
tracks token usage and the stop reason for the metrics log:

* :class:`Passthrough` forwards bytes unchanged (same API on both sides) and parses a copy;
* :class:`OpenAIToAnthropic` turns ``chat.completion.chunk`` events into Anthropic
  ``message_start`` / ``content_block_*`` / ``message_delta`` / ``message_stop`` events,
  including streamed tool calls (``input_json_delta``);
* :class:`AnthropicToOpenAI` does the reverse, ending with ``data: [DONE]``.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from agent_model_router.sse import SSEEvent
from agent_model_router.sse import SSEParser
from agent_model_router.sse import encode_event
from agent_model_router.translate.responses import ANTHROPIC_STOP_TO_OPENAI
from agent_model_router.translate.responses import OPENAI_FINISH_TO_ANTHROPIC
from agent_model_router.translate.responses import Usage

__all__ = ["AnthropicToOpenAI", "OpenAIToAnthropic", "Passthrough", "StreamAdapter", "make_adapter"]


def _loads(event: SSEEvent) -> dict[str, Any] | None:
    if event.data == "[DONE]":
        return None
    try:
        value = json.loads(event.data)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class StreamAdapter:
    """Base: parses upstream SSE and tracks usage / stop reason (Anthropic vocabulary)."""

    def __init__(self) -> None:
        self.parser = SSEParser()
        self.usage = Usage()
        self.stop_reason: str | None = None
        self.text_chars = 0
        self.error: str | None = None

    def feed(self, chunk: bytes) -> bytes:
        return b"".join(self.on_event(e) for e in self.parser.feed(chunk))

    def finish(self) -> bytes:
        out = b"".join(self.on_event(e) for e in self.parser.flush())
        return out + self.on_end()

    def on_event(self, event: SSEEvent) -> bytes:  # pragma: no cover - overridden
        raise NotImplementedError

    def on_end(self) -> bytes:
        return b""

    # usage helpers
    def _anthropic_observe(self, data: dict[str, Any]) -> None:
        kind = data.get("type")
        if kind == "message_start":
            u = (data.get("message") or {}).get("usage") or {}
            self.usage.input_tokens = (
                int(u.get("input_tokens") or 0)
                + int(u.get("cache_read_input_tokens") or 0)
                + int(u.get("cache_creation_input_tokens") or 0)
            )
            self.usage.output_tokens = int(u.get("output_tokens") or 0)
            self.usage.known = True
        elif kind == "message_delta":
            u = data.get("usage") or {}
            if "output_tokens" in u:
                self.usage.output_tokens = int(u.get("output_tokens") or 0)
                self.usage.known = True
            reason = (data.get("delta") or {}).get("stop_reason")
            if reason:
                self.stop_reason = str(reason)
        elif kind == "content_block_delta":
            delta = data.get("delta") or {}
            self.text_chars += len(str(delta.get("text") or delta.get("partial_json") or ""))
        elif kind == "error":
            err = data.get("error") or {}
            self.error = str(err.get("type") or "error")

    def _openai_observe(self, data: dict[str, Any]) -> None:
        u = data.get("usage")
        if isinstance(u, dict):
            self.usage.input_tokens = int(u.get("prompt_tokens") or 0)
            self.usage.output_tokens = int(u.get("completion_tokens") or 0)
            self.usage.known = True
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            self.text_chars += len(str(delta.get("content") or ""))
            for call in delta.get("tool_calls") or []:
                self.text_chars += len(str((call.get("function") or {}).get("arguments") or ""))
            if choice.get("finish_reason"):
                self.stop_reason = OPENAI_FINISH_TO_ANTHROPIC.get(
                    choice["finish_reason"], "end_turn"
                )
        if isinstance(data.get("error"), dict):
            self.error = "error"


class Passthrough(StreamAdapter):
    """Same API both ways: bytes go out exactly as they came in."""

    def __init__(self, api: str) -> None:
        super().__init__()
        self.api = api

    def feed(self, chunk: bytes) -> bytes:
        for event in self.parser.feed(chunk):
            self.on_event(event)
        return chunk

    def finish(self) -> bytes:
        for event in self.parser.flush():
            self.on_event(event)
        return b""

    def on_event(self, event: SSEEvent) -> bytes:
        data = _loads(event)
        if data is not None:
            if self.api == "anthropic":
                self._anthropic_observe(data)
            else:
                self._openai_observe(data)
        return b""


class OpenAIToAnthropic(StreamAdapter):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model
        self.started = False
        self.block_index = -1
        self.open_block: str | None = None  # "text" | "tool:<openai index>"
        self.tool_blocks: dict[int, int] = {}
        self.finish_reason: str | None = None
        self.ended = False

    def _start(self) -> bytes:
        if self.started:
            return b""
        self.started = True
        message: dict[str, Any] = {
            "id": "msg_" + uuid.uuid4().hex[:24],
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return encode_event({"type": "message_start", "message": message}, "message_start")

    def _close(self) -> bytes:
        if self.open_block is None:
            return b""
        self.open_block = None
        return encode_event(
            {"type": "content_block_stop", "index": self.block_index}, "content_block_stop"
        )

    def _open(self, key: str, block: dict[str, Any]) -> bytes:
        out = self._close()
        self.block_index += 1
        self.open_block = key
        return out + encode_event(
            {"type": "content_block_start", "index": self.block_index, "content_block": block},
            "content_block_start",
        )

    def on_event(self, event: SSEEvent) -> bytes:
        data = _loads(event)
        if data is None:
            return b""
        self._openai_observe(data)
        out = self._start()
        if isinstance(data.get("error"), dict):
            msg = str(data["error"].get("message") or "upstream error")[:300]
            return out + encode_event(
                {"type": "error", "error": {"type": "api_error", "message": msg}}, "error"
            )
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                if self.open_block != "text":
                    out += self._open("text", {"type": "text", "text": ""})
                out += encode_event(
                    {
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                    "content_block_delta",
                )
            for call in delta.get("tool_calls") or []:
                idx = int(call.get("index") or 0)
                fn = call.get("function") or {}
                key = f"tool:{idx}"
                if idx not in self.tool_blocks:
                    out += self._open(
                        key,
                        {
                            "type": "tool_use",
                            "id": str(call.get("id") or "toolu_" + uuid.uuid4().hex[:24]),
                            "name": str(fn.get("name") or ""),
                            "input": {},
                        },
                    )
                    self.tool_blocks[idx] = self.block_index
                elif self.open_block != key:
                    # Interleaved fragments for an earlier call cannot be re-opened in the
                    # Anthropic stream; they are dropped (servers do not do this in practice).
                    continue
                args = fn.get("arguments")
                if args:
                    out += encode_event(
                        {
                            "type": "content_block_delta",
                            "index": self.block_index,
                            "delta": {"type": "input_json_delta", "partial_json": args},
                        },
                        "content_block_delta",
                    )
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])
        return out

    def on_end(self) -> bytes:
        if self.ended:
            return b""
        self.ended = True
        out = self._start() + self._close()
        stop = OPENAI_FINISH_TO_ANTHROPIC.get(self.finish_reason or "stop", "end_turn")
        if self.tool_blocks and stop == "end_turn":
            stop = "tool_use"
        self.stop_reason = stop
        out += encode_event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                },
            },
            "message_delta",
        )
        return out + encode_event({"type": "message_stop"}, "message_stop")


class AnthropicToOpenAI(StreamAdapter):
    def __init__(self, model: str, include_usage: bool = False) -> None:
        super().__init__()
        self.model = model
        self.include_usage = include_usage
        self.id = "chatcmpl-" + uuid.uuid4().hex[:24]
        self.created = int(time.time())
        self.tool_index: dict[int, int] = {}
        self.done = False

    def _chunk(self, delta: dict[str, Any], finish: str | None = None) -> bytes:
        return encode_event(
            {
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    def on_event(self, event: SSEEvent) -> bytes:
        data = _loads(event)
        if data is None:
            return b""
        self._anthropic_observe(data)
        kind = data.get("type")
        if kind == "message_start":
            return self._chunk({"role": "assistant", "content": ""})
        if kind == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                idx = len(self.tool_index)
                self.tool_index[int(data.get("index") or 0)] = idx
                return self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": idx,
                                "id": str(block.get("id")),
                                "type": "function",
                                "function": {"name": str(block.get("name")), "arguments": ""},
                            }
                        ]
                    }
                )
            return b""
        if kind == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                return self._chunk({"content": str(delta.get("text", ""))})
            if delta.get("type") == "input_json_delta":
                tool_idx = self.tool_index.get(int(data.get("index") or 0))
                if tool_idx is None:
                    return b""
                return self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": tool_idx,
                                "function": {"arguments": str(delta.get("partial_json", ""))},
                            }
                        ]
                    }
                )
            return b""  # thinking / signature deltas have no Chat Completions counterpart
        if kind == "message_delta":
            reason = str((data.get("delta") or {}).get("stop_reason") or "end_turn")
            return self._chunk({}, ANTHROPIC_STOP_TO_OPENAI.get(reason, "stop"))
        if kind == "error":
            err = data.get("error") or {}
            return encode_event(
                {
                    "error": {
                        "message": str(err.get("message", "upstream error"))[:300],
                        "type": str(err.get("type", "api_error")),
                    }
                }
            )
        if kind == "message_stop":
            return self._done()
        return b""

    def _done(self) -> bytes:
        if self.done:
            return b""
        self.done = True
        out = b""
        if self.include_usage:
            out += encode_event(
                {
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": self.usage.input_tokens,
                        "completion_tokens": self.usage.output_tokens,
                        "total_tokens": self.usage.input_tokens + self.usage.output_tokens,
                    },
                }
            )
        return out + encode_event("[DONE]")

    def on_end(self) -> bytes:
        return self._done()


def make_adapter(
    inbound_api: str, backend_api: str, model: str, include_usage: bool = False
) -> StreamAdapter:
    if inbound_api == backend_api:
        return Passthrough(inbound_api)
    if inbound_api == "anthropic":
        return OpenAIToAnthropic(model)
    return AnthropicToOpenAI(model, include_usage=include_usage)
