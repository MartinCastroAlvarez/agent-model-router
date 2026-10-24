"""Request translation between Anthropic Messages and OpenAI Chat Completions.

Covers text, images, system prompts, tools, tool calls and tool results, tool_choice,
stop sequences, sampling and streaming flags. Provider-specific fields that have no
counterpart (cache_control, thinking blocks, metadata) are dropped on the way across.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["anthropic_to_openai_request", "openai_to_anthropic_request"]

DEFAULT_MAX_TOKENS = 4096


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


# --------------------------------------------------------------------------- #
# Anthropic -> OpenAI
# --------------------------------------------------------------------------- #
def _anthropic_image_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    elif source.get("type") == "url":
        url = str(source.get("url", ""))
    else:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            elif isinstance(b, dict) and b.get("type") == "image":
                parts.append("[image]")
        return "\n".join(parts)
    return "" if content is None else json.dumps(content)


def _anthropic_messages_to_openai(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        blocks = [b for b in content or [] if isinstance(b, dict)]
        if role == "assistant":
            text = "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
            calls = [
                {
                    "id": str(b.get("id")),
                    "type": "function",
                    "function": {
                        "name": str(b.get("name")),
                        "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                    },
                }
                for b in blocks
                if b.get("type") == "tool_use"
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue
        # user (or a system role mid-conversation): tool results become role=tool messages
        for b in blocks:
            if b.get("type") == "tool_result":
                text = _tool_result_text(b.get("content"))
                if b.get("is_error"):
                    text = "[tool error] " + text
                out.append(
                    {"role": "tool", "tool_call_id": str(b.get("tool_use_id")), "content": text}
                )
        parts: list[dict[str, Any]] = []
        for b in blocks:
            if b.get("type") == "text":
                parts.append({"type": "text", "text": str(b.get("text", ""))})
            elif b.get("type") == "image":
                image = _anthropic_image_to_openai(b)
                if image:
                    parts.append(image)
        if parts:
            if all(p["type"] == "text" for p in parts):
                out.append({"role": role, "content": "\n".join(p["text"] for p in parts)})
            else:
                out.append({"role": role, "content": parts})
    return out


def _anthropic_tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        return {"type": "function", "function": {"name": choice.get("name")}}
    return None


def anthropic_to_openai_request(body: dict[str, Any], model: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = _text_of(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    messages += _anthropic_messages_to_openai(list(body.get("messages") or []))
    out: dict[str, Any] = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), int):
        out["max_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p"):
        if key in body:
            out[key] = body[key]
    if body.get("stop_sequences"):
        out["stop"] = list(body["stop_sequences"])
    tools = [
        {
            "type": "function",
            "function": {
                "name": t.get("name"),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for t in body.get("tools") or []
        if isinstance(t, dict) and t.get("type") in (None, "custom") and t.get("name")
    ]
    if tools:
        out["tools"] = tools
        choice = _anthropic_tool_choice(body.get("tool_choice"))
        if choice is not None:
            out["tool_choice"] = choice
        if isinstance(body.get("tool_choice"), dict) and body["tool_choice"].get(
            "disable_parallel_tool_use"
        ):
            out["parallel_tool_calls"] = False
    if body.get("stream"):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


# --------------------------------------------------------------------------- #
# OpenAI -> Anthropic
# --------------------------------------------------------------------------- #
def _openai_part_to_anthropic(part: dict[str, Any]) -> dict[str, Any] | None:
    if part.get("type") == "text":
        return {"type": "text", "text": str(part.get("text", ""))}
    if part.get("type") == "image_url":
        image = part.get("image_url") or {}
        url = str(image.get("url", "") if isinstance(image, dict) else image)
        if url.startswith("data:") and ";base64," in url:
            head, data = url.split(";base64,", 1)
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": head[5:], "data": data},
            }
        if url.startswith("https://"):
            return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _openai_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for part in content or []:
        if isinstance(part, dict):
            block = _openai_part_to_anthropic(part)
            if block:
                blocks.append(block)
    return blocks


def _parse_arguments(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {"_raw_arguments": str(raw)}
    return value if isinstance(value, dict) else {"value": value}


def _push(out: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    """Append, merging consecutive same-role turns (Anthropic wants alternation)."""
    if not blocks:
        return
    if out and out[-1]["role"] == role:
        out[-1]["content"].extend(blocks)
    else:
        out.append({"role": role, "content": blocks})


def openai_to_anthropic_request(body: dict[str, Any], model: str) -> dict[str, Any]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in {"system", "developer"}:
            system_parts.append(_text_of(msg.get("content")))
        elif role == "user":
            _push(messages, "user", _openai_content_blocks(msg.get("content")))
        elif role == "assistant":
            blocks = _openai_content_blocks(msg.get("content"))
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(call.get("id")),
                        "name": str(fn.get("name")),
                        "input": _parse_arguments(fn.get("arguments")),
                    }
                )
            _push(messages, "assistant", blocks)
        elif role == "tool":
            _push(
                messages,
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": str(msg.get("tool_call_id")),
                        "content": _text_of(msg.get("content")),
                    }
                ],
            )
    out: dict[str, Any] = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n\n".join(p for p in system_parts if p)
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    out["max_tokens"] = max_tokens if isinstance(max_tokens, int) else DEFAULT_MAX_TOKENS
    for key in ("temperature", "top_p"):
        if key in body:
            out[key] = body[key]
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    tools = []
    for t in body.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            tools.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
    if tools:
        out["tools"] = tools
        choice = body.get("tool_choice")
        mapped: dict[str, Any] | None = None
        if choice == "auto":
            mapped = {"type": "auto"}
        elif choice == "required":
            mapped = {"type": "any"}
        elif choice == "none":
            mapped = {"type": "none"}
        elif isinstance(choice, dict) and isinstance(choice.get("function"), dict):
            mapped = {"type": "tool", "name": choice["function"].get("name")}
        if body.get("parallel_tool_calls") is False:
            mapped = {**(mapped or {"type": "auto"}), "disable_parallel_tool_use": True}
        if mapped:
            out["tool_choice"] = mapped
    if body.get("stream"):
        out["stream"] = True
    return out
