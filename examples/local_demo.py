"""End-to-end demo on real sockets, no models and no network.

Starts the fake OpenAI-compatible backend and the router (uvicorn, loopback, random ports),
then sends an Anthropic request (as Claude Code would), an OpenAI request, a streaming
request and an explain call, and prints the routes and the savings report.

    python -m examples.local_demo
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from agent_model_router.config import parse_config
from agent_model_router.metrics import MetricsStore
from agent_model_router.metrics import format_stats
from agent_model_router.proxy import create_app
from examples.fake_backend import app as fake_app

TOKEN = "amr-demo-token-not-a-secret"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def serve(app: object, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.05)
    return server


def main() -> None:
    backend_port, router_port = free_port(), free_port()
    workdir = Path(tempfile.mkdtemp(prefix="amr-demo-"))
    config = parse_config(
        {
            "backends": {
                "ollama": {"kind": "ollama", "base_url": f"http://127.0.0.1:{backend_port}/v1"}
            },
            "models": [
                {
                    "name": "small",
                    "backend": "ollama",
                    "model": "qwen3:1.7b",
                    "quality": 0.3,
                    "price": {"input": 0.01, "output": 0.02},
                    "effort": "reasoning_effort",
                },
                {
                    "name": "medium",
                    "backend": "ollama",
                    "model": "qwen3:8b",
                    "quality": 0.55,
                    "price": {"input": 0.05, "output": 0.10},
                    "effort": "reasoning_effort",
                },
                {
                    "name": "large",
                    "backend": "ollama",
                    "model": "qwen3-coder:30b",
                    "quality": 0.8,
                    "price": {"input": 0.20, "output": 0.40},
                    "context": 262144,
                },
            ],
            "metrics": {"db": str(workdir / "routes.sqlite3")},
        }
    )
    metrics = MetricsStore(config.metrics_db)
    servers = [
        serve(fake_app, backend_port),
        serve(create_app(config, token=TOKEN, metrics=metrics), router_port),
    ]
    base = f"http://127.0.0.1:{router_port}"
    prompts = [
        ("anthropic", "Rename get_usr to get_user in utils/users.py"),
        (
            "openai",
            "Review this pull request for the invoices feature: naming, test coverage "
            "and error handling.",
        ),
        (
            "anthropic",
            "Refactor the scheduler to be thread-safe across the entire codebase and "
            "remove the race condition in the async worker pool.",
        ),
    ]
    with httpx.Client(base_url=base, headers={"x-api-key": TOKEN}, timeout=10) as client:
        for api, text in prompts:
            path = "/v1/messages" if api == "anthropic" else "/v1/chat/completions"
            body = {
                "model": "auto",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": text}],
            }
            r = client.post(path, json=body)
            r.raise_for_status()
            reply = (
                r.json()["content"][0]["text"]
                if api == "anthropic"
                else r.json()["choices"][0]["message"]["content"]
            )
            print(
                f"{api:<9} -> {r.headers['x-router-model']:<7} "
                f"({r.headers['x-router-bucket']}, effort {r.headers['x-router-effort']}) "
                f"reply: {reply!r}"
            )

        body = {
            "model": "auto",
            "max_tokens": 512,
            "stream": True,
            "messages": [{"role": "user", "content": prompts[0][1]}],
        }
        with client.stream("POST", "/v1/messages", json=body) as r:
            kinds = [
                json.loads(line[6:])["type"] for line in r.iter_lines() if line.startswith("data: ")
            ]
        print("stream    -> Anthropic events:", " ".join(dict.fromkeys(kinds)))
        assert kinds[0] == "message_start" and kinds[-1] == "message_stop"

        explained = client.post(
            "/router/explain", json={"messages": [{"role": "user", "content": prompts[2][1]}]}
        ).json()
        print("explain   ->", explained["model"], "|", explained["reason"])
        print("             top features:", explained["decision"]["contributions"][:3])

    for server in servers:
        server.should_exit = True
    time.sleep(0.3)
    print()
    print(format_stats(metrics.summary(), config.baseline().name))


if __name__ == "__main__":
    main()
