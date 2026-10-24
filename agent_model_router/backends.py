"""Upstream calls: build the backend request (dialect, model, effort) and send it.

One ``httpx.AsyncClient`` per backend, created from the config only: no redirects are
followed (a key must not follow a redirect to another host), proxy environment variables
are ignored, and headers come from an allowlist (:func:`security.upstream_headers`).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import httpx

from agent_model_router.config import Backend
from agent_model_router.config import Config
from agent_model_router.config import Model
from agent_model_router.config import Policy
from agent_model_router.security import upstream_headers
from agent_model_router.translate.requests import DEFAULT_MAX_TOKENS
from agent_model_router.translate.requests import anthropic_to_openai_request
from agent_model_router.translate.requests import openai_to_anthropic_request

__all__ = ["Upstream", "apply_effort", "prepare_body"]

_SAMPLING = ("temperature", "top_p", "top_k")


def _strip_reasoning(body: dict[str, Any]) -> None:
    body.pop("thinking", None)
    body.pop("reasoning_effort", None)
    body.pop("reasoning", None)
    oc = body.get("output_config")
    if isinstance(oc, dict):
        oc = {k: v for k, v in oc.items() if k != "effort"}
        if oc:
            body["output_config"] = oc
        else:
            body.pop("output_config")


def apply_effort(
    body: dict[str, Any], api: str, model: Model, effort: str, policy: Policy
) -> dict[str, Any]:
    """Set the reasoning effort the router chose, replacing whatever the client sent.

    Styles: ``reasoning_effort`` (OpenAI-compatible servers, ollama), ``adaptive``
    (``thinking: {type: adaptive}`` + ``output_config.effort``, current Claude models),
    ``budget`` (``thinking.budget_tokens``, older Claude models), ``none`` (strip).
    """
    _strip_reasoning(body)
    style = model.effort_style
    if style == "none":
        return body
    if api == "openai":
        if effort != "none":
            body["reasoning_effort"] = effort
        return body
    forced_tool = isinstance(body.get("tool_choice"), dict) and body["tool_choice"].get("type") in {
        "any",
        "tool",
    }
    if style in {"adaptive", "reasoning_effort"}:
        body["thinking"] = {"type": "adaptive"}
        body.setdefault("output_config", {})
        body["output_config"]["effort"] = "low" if effort == "none" else effort
        for key in _SAMPLING:
            body.pop(key, None)
        return body
    # budget style
    if effort == "none" or forced_tool:
        return body
    budget = int(policy.thinking_budgets.get(effort, 4096))
    max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_TOKENS)
    if max_tokens <= budget:
        max_tokens = min(model.max_output, budget + max(1024, max_tokens))
    if max_tokens <= budget:
        budget = max_tokens - 1024
    if budget < 1024:
        return body
    body["max_tokens"] = max_tokens
    body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    for key in _SAMPLING:
        body.pop(key, None)
    return body


def prepare_body(
    inbound_api: str,
    body: Mapping[str, Any],
    model: Model,
    backend: Backend,
    effort: str,
    policy: Policy,
) -> dict[str, Any]:
    """The JSON body for ``backend``: translated if needed, model swapped, effort applied."""
    if inbound_api == backend.api:
        out = copy.deepcopy(dict(body))
        out["model"] = model.model
    elif backend.api == "openai":
        out = anthropic_to_openai_request(dict(body), model.model)
    else:
        out = openai_to_anthropic_request(dict(body), model.model)
    if backend.api == "anthropic":
        out["max_tokens"] = min(int(out.get("max_tokens") or DEFAULT_MAX_TOKENS), model.max_output)
    else:
        for key in ("max_tokens", "max_completion_tokens"):
            if isinstance(out.get(key), int):
                out[key] = min(out[key], model.max_output)
    return apply_effort(out, backend.api, model, effort, policy)


class Upstream:
    """HTTP clients for the configured backends."""

    def __init__(self, config: Config, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.clients: dict[str, httpx.AsyncClient] = {}
        for name, backend in config.backends.items():
            self.clients[name] = httpx.AsyncClient(
                base_url=backend.base_url,
                timeout=httpx.Timeout(backend.timeout_s, connect=backend.connect_timeout_s),
                follow_redirects=False,
                trust_env=False,
                transport=transport,
            )

    @staticmethod
    def path(backend: Backend) -> str:
        return "/v1/messages" if backend.api == "anthropic" else "/chat/completions"

    def build(
        self,
        backend: Backend,
        body: dict[str, Any],
        inbound_headers: Mapping[str, str],
        inbound_api: str,
        *,
        path: str | None = None,
    ) -> httpx.Request:
        headers = upstream_headers(
            inbound_headers,
            inbound_api=inbound_api,
            backend_api=backend.api,
            api_key=backend.api_key(),
            extra=backend.headers,
        )
        if body.get("stream"):
            headers["accept"] = "text/event-stream"
        client = self.clients[backend.name]
        return client.build_request("POST", path or self.path(backend), json=body, headers=headers)

    async def send(self, backend: Backend, request: httpx.Request, stream: bool) -> httpx.Response:
        return await self.clients[backend.name].send(request, stream=stream)

    async def complete_text(
        self, model: Model, messages: list[dict[str, str]], max_tokens: int = 200
    ) -> str:
        """A small non-streaming completion (used by the decision-model classifier)."""
        backend = self.config.backends[model.backend]
        body = prepare_body(
            "openai",
            {"messages": messages, "max_tokens": max_tokens},
            model,
            backend,
            "none",
            self.config.policy,
        )
        request = self.build(backend, body, {}, "openai")
        response = await self.send(backend, request, stream=False)
        response.raise_for_status()
        data = response.json()
        if backend.api == "anthropic":
            return "".join(
                b.get("text", "")
                for b in data.get("content") or []
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")

    async def aclose(self) -> None:
        for client in self.clients.values():
            await client.aclose()
