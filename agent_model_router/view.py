"""A protocol-neutral view of a chat request, used by the classifier and the policy.

Both inbound dialects (Anthropic Messages, OpenAI Chat Completions) are reduced to the
few facts routing needs: the latest human text, earlier human turns, whether the request
is a tool-loop step, tool presence and an estimate of the context size. The view holds
text only in memory; nothing here is logged or stored.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any

__all__ = ["RequestView", "estimate_tokens", "view_request"]

_IMAGE_TOKENS = 1200


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 characters per token), good enough for routing."""
    return (len(text) + 3) // 4


@dataclass(frozen=True)
class RequestView:
    api: str
    last_user_text: str
    earlier_user_texts: tuple[str, ...]
    last_assistant_text: str
    system_text: str
    tools_present: bool
    tool_followup: bool
    user_turns: int
    recent_tool_results: int
    context_tokens: int
    max_tokens: int | None
    stream: bool
    requested_model: str | None
    session_hint: str | None = None
    images: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def session_key(self, headers: Mapping[str, str]) -> str:
        """A stable, hashed conversation id.

        Explicit ids win (Claude Code's ``X-Claude-Code-Session-Id``, ``x-session-id``,
        the body's ``metadata.user_id`` / ``user``); otherwise a hash of the system prompt
        prefix and the first human message, which stays constant through a conversation.
        """
        lowered = {k.lower(): v for k, v in headers.items()}
        explicit = (
            lowered.get("x-claude-code-session-id")
            or lowered.get("x-session-id")
            or self.session_hint
        )
        if explicit:
            material = "id:" + explicit
        else:
            first = self.earlier_user_texts[0] if self.earlier_user_texts else self.last_user_text
            material = "conv:" + self.system_text[:2000] + "\x00" + first[:4000]
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _blocks_text(content: Any, *, skip_tool_results: bool = True) -> tuple[str, int, int]:
    """(text, tool_result_count, image_count) for Anthropic or OpenAI content."""
    if isinstance(content, str):
        return content, 0, 0
    if not isinstance(content, list):
        return "", 0, 0
    parts: list[str] = []
    tool_results = 0
    images = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif kind == "tool_result":
            tool_results += 1
            if not skip_tool_results:
                parts.append(_blocks_text(block.get("content"))[0])
        elif kind in {"image", "image_url", "document"}:
            images += 1
    return "\n".join(parts), tool_results, images


def _size(obj: Any) -> int:
    return estimate_tokens(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def _system_text(system: Any) -> str:
    text, _, _ = _blocks_text(system)
    return text


def _anthropic_view(body: Mapping[str, Any]) -> RequestView:
    messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    human: list[str] = []
    last_assistant = ""
    tool_followup = False
    recent_tool_results = 0
    images = 0
    for i, msg in enumerate(messages):
        text, n_results, n_images = _blocks_text(msg.get("content"))
        images += n_images
        if msg.get("role") == "user":
            if text.strip():
                human.append(text)
            if i == len(messages) - 1:
                tool_followup = n_results > 0 and not text.strip()
            if i >= len(messages) - 6:
                recent_tool_results += n_results
        elif msg.get("role") == "assistant":
            last_assistant = text
    raw_meta = body.get("metadata")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    hint = meta.get("user_id") if isinstance(meta.get("user_id"), str) else None
    return RequestView(
        api="anthropic",
        last_user_text=human[-1] if human else "",
        earlier_user_texts=tuple(human[:-1]),
        last_assistant_text=last_assistant,
        system_text=_system_text(body.get("system")),
        tools_present=bool(body.get("tools")),
        tool_followup=tool_followup,
        user_turns=len(human),
        recent_tool_results=recent_tool_results,
        context_tokens=_size(body.get("messages"))
        + _size(body.get("system") or "")
        + _size(body.get("tools") or [])
        + images * _IMAGE_TOKENS,
        max_tokens=body.get("max_tokens") if isinstance(body.get("max_tokens"), int) else None,
        stream=bool(body.get("stream")),
        requested_model=body.get("model") if isinstance(body.get("model"), str) else None,
        session_hint=hint,
        images=images,
    )


def _openai_view(body: Mapping[str, Any]) -> RequestView:
    messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    human: list[str] = []
    system_parts: list[str] = []
    last_assistant = ""
    recent_tool_results = 0
    images = 0
    for i, msg in enumerate(messages):
        role = msg.get("role")
        text, _, n_images = _blocks_text(msg.get("content"))
        images += n_images
        if role in {"system", "developer"}:
            system_parts.append(text)
        elif role == "user" and text.strip():
            human.append(text)
        elif role == "assistant":
            last_assistant = text
        elif role == "tool" and i >= len(messages) - 6:
            recent_tool_results += 1
    tool_followup = bool(messages) and messages[-1].get("role") == "tool"
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    hint = body.get("user") if isinstance(body.get("user"), str) else None
    return RequestView(
        api="openai",
        last_user_text=human[-1] if human else "",
        earlier_user_texts=tuple(human[:-1]),
        last_assistant_text=last_assistant,
        system_text="\n".join(system_parts),
        tools_present=bool(body.get("tools")),
        tool_followup=tool_followup,
        user_turns=len(human),
        recent_tool_results=recent_tool_results,
        context_tokens=_size(messages) + _size(body.get("tools") or []) + images * _IMAGE_TOKENS,
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        stream=bool(body.get("stream")),
        requested_model=body.get("model") if isinstance(body.get("model"), str) else None,
        session_hint=hint,
        images=images,
    )


def view_request(api: str, body: Mapping[str, Any]) -> RequestView:
    """Build a :class:`RequestView` for an ``anthropic`` or ``openai`` request body."""
    if api == "anthropic":
        return _anthropic_view(body)
    if api == "openai":
        return _openai_view(body)
    raise ValueError(f"unknown api {api!r}")
