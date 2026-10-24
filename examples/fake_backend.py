"""A fake OpenAI-compatible model server (stand-in for ollama when it is not installed).

It answers ``POST /v1/chat/completions`` (JSON or SSE) with a canned reply that names the
model it was asked for, so the demo shows where each request was routed. No model runs.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import Response
from starlette.responses import StreamingResponse
from starlette.routing import Route


async def completions(request: Request) -> Response:
    body = await request.json()
    model = body.get("model", "?")
    effort = body.get("reasoning_effort", "none")
    text = f"[{model} effort={effort}] Done."
    if not body.get("stream"):
        return JSONResponse(
            {
                "id": "fake-1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 12},
            }
        )

    async def events() -> AsyncIterator[bytes]:
        base = {"id": "fake-2", "object": "chat.completion.chunk", "created": 0, "model": model}
        for word in text.split(" "):
            chunk = {**base, "choices": [{"index": 0, "delta": {"content": word + " "}}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        done = {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 12},
        }
        yield f"data: {json.dumps(done)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


app = Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"])])
