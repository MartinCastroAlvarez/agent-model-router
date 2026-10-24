"""Incremental Server-Sent Events parsing and encoding.

Upstream chunks can split an event anywhere (even inside a UTF-8 sequence or a CRLF);
:class:`SSEParser` buffers bytes and yields complete events only.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

__all__ = ["SSEEvent", "SSEParser", "encode_event"]


@dataclass(frozen=True)
class SSEEvent:
    event: str | None
    data: str

    def json(self) -> Any:
        return json.loads(self.data)


class SSEParser:
    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes) -> Iterator[SSEEvent]:
        self._buffer += chunk
        self._buffer = self._buffer.replace(b"\r\n", b"\n")
        # A lone trailing \r may be the first half of a CRLF; keep it for the next chunk.
        while True:
            idx = self._buffer.find(b"\n\n")
            if idx < 0:
                break
            raw, self._buffer = self._buffer[:idx], self._buffer[idx + 2 :]
            event = self._parse(raw.replace(b"\r", b"\n").decode("utf-8", "replace"))
            if event is not None:
                yield event

    def flush(self) -> Iterator[SSEEvent]:
        """Emit a final event that was not terminated by a blank line."""
        raw, self._buffer = self._buffer, b""
        if raw.strip():
            event = self._parse(raw.decode("utf-8", "replace"))
            if event is not None:
                yield event

    @staticmethod
    def _parse(raw: str) -> SSEEvent | None:
        event: str | None = None
        data: list[str] = []
        for line in raw.split("\n"):
            if not line or line.startswith(":"):
                continue
            name, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if name == "event":
                event = value
            elif name == "data":
                data.append(value)
        if event is None and not data:
            return None
        return SSEEvent(event=event, data="\n".join(data))


def encode_event(data: Any, event: str | None = None) -> bytes:
    """One SSE event; ``data`` is JSON-encoded unless it is already a string."""
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    head = f"event: {event}\n" if event else ""
    return (head + f"data: {payload}\n\n").encode("utf-8")
