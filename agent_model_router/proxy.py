"""The Starlette app: inbound Anthropic + OpenAI endpoints, routing, escalation, metrics.

Endpoints:

* ``POST /v1/messages`` (Anthropic Messages; Claude Code via ``ANTHROPIC_BASE_URL``)
* ``POST /v1/messages/count_tokens`` (estimate)
* ``POST /v1/chat/completions`` (OpenAI Chat Completions; any OpenAI-compatible client)
* ``GET /v1/models`` (the pool, plus ``auto``)
* ``POST /router/explain`` (dry run: the decision and route, nothing is forwarded)
* ``GET /router/stats`` (the metrics summary)
* ``GET /healthz`` (no auth, no details)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import Response
from starlette.responses import StreamingResponse
from starlette.routing import Route as StarletteRoute

from agent_model_router import __version__
from agent_model_router.backends import Upstream
from agent_model_router.backends import prepare_body
from agent_model_router.classifier.decision import DecisionModelClassifier
from agent_model_router.classifier.heuristic import Classifier
from agent_model_router.classifier.heuristic import HeuristicClassifier
from agent_model_router.classifier.linear import LinearScorer
from agent_model_router.config import Config
from agent_model_router.config import Model
from agent_model_router.metrics import MetricsStore
from agent_model_router.metrics import RouteRecord
from agent_model_router.policy import NoRoute
from agent_model_router.policy import Route
from agent_model_router.policy import SessionStore
from agent_model_router.policy import estimate_cost
from agent_model_router.policy import record
from agent_model_router.policy import select
from agent_model_router.security import UPSTREAM_RESPONSE_HEADERS
from agent_model_router.security import is_loopback_host
from agent_model_router.security import tokens_match
from agent_model_router.translate.responses import Usage
from agent_model_router.translate.responses import anthropic_to_openai_response
from agent_model_router.translate.responses import error_for
from agent_model_router.translate.responses import openai_to_anthropic_response
from agent_model_router.translate.responses import upstream_error_message
from agent_model_router.translate.responses import usage_from
from agent_model_router.translate.streams import make_adapter
from agent_model_router.view import RequestView
from agent_model_router.view import view_request

__all__ = ["RouterState", "create_app"]

log = logging.getLogger("agent_model_router")

RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_REFUSAL_PREFIXES = (
    "i can't help with",
    "i cannot help with",
    "i can't assist",
    "i cannot assist",
    "i'm not able to help",
    "i am not able to help",
    "i won't help",
    "i'm sorry, but i can't",
    "sorry, i can't",
    "i must decline",
)


def _looks_refused(api: str, body: dict[str, Any]) -> bool:
    """A refusal stop reason, or a short reply that opens with a refusal phrase."""
    if api == "anthropic":
        if body.get("stop_reason") == "refusal":
            return True
        text = "".join(
            b.get("text", "")
            for b in body.get("content") or []
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        choice = (body.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "content_filter":
            return True
        text = str((choice.get("message") or {}).get("content") or "")
    text = text.strip().lower().replace("’", "'")
    return len(text) < 400 and text.startswith(_REFUSAL_PREFIXES)


def _host_of(header: str) -> str:
    """The hostname of a Host header (``[::1]:8787`` -> ``::1``)."""
    if header.startswith("["):
        return header[1 : header.find("]")]
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def _ascii(value: str, limit: int = 200) -> str:
    return value.encode("ascii", "replace").decode("ascii")[:limit]


@dataclass
class RouterState:
    config: Config
    upstream: Upstream
    classifier: Classifier
    sessions: SessionStore
    metrics: MetricsStore
    token: str | None
    clock: Callable[[], float]


def _build_classifier(config: Config, upstream: Upstream) -> Classifier:
    scorer = LinearScorer.load(config.classifier.weights)
    heuristic = HeuristicClassifier(scorer)
    if config.classifier.mode == "decision" and config.classifier.decision_model:
        judge = config.model(config.classifier.decision_model)

        async def ask(messages: list[dict[str, str]]) -> str:
            return await upstream.complete_text(judge, messages)

        return DecisionModelClassifier(
            ask, heuristic=heuristic, timeout_s=config.classifier.decision_timeout_s
        )
    return heuristic


class Router:
    def __init__(self, state: RouterState) -> None:
        self.s = state

    # ------------------------------------------------------------------ guards
    def guard(self, request: Request, api: str, *, auth: bool = True) -> Response | None:
        server = self.s.config.server
        host = _host_of(request.headers.get("host") or "")
        if is_loopback_host(server.host) and not is_loopback_host(host):
            # DNS rebinding: a page on evil.example resolving to 127.0.0.1 sends its own Host.
            return JSONResponse(error_for(api, 403, "host not allowed"), status_code=403)
        origin = request.headers.get("origin")
        if origin and origin not in server.allowed_origins:
            return JSONResponse(error_for(api, 403, "origin not allowed"), status_code=403)
        if auth and self.s.token is not None:
            presented = request.headers.get("x-api-key")
            authz = request.headers.get("authorization", "")
            if not presented and authz.lower().startswith("bearer "):
                presented = authz[7:].strip()
            if not tokens_match(presented, self.s.token):
                status = 401
                return JSONResponse(
                    error_for(api, status, "invalid router token"), status_code=status
                )
        return None

    async def read_json(self, request: Request, api: str) -> dict[str, Any] | Response:
        limit = self.s.config.server.max_body_bytes
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            return JSONResponse(error_for(api, 413, "request too large"), status_code=413)
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                return JSONResponse(error_for(api, 413, "request too large"), status_code=413)
            chunks.append(chunk)
        try:
            body = json.loads(b"".join(chunks) or b"null")
        except ValueError:
            return JSONResponse(error_for(api, 400, "body must be JSON"), status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return JSONResponse(
                error_for(api, 400, "body must be an object with messages"), status_code=400
            )
        return body

    # ------------------------------------------------------------------ routing
    def _budget_exceeded(self) -> bool:
        daily = self.s.config.budgets.daily_usd
        if daily is None:
            return False
        now = self.s.clock()
        local = time.localtime(now)
        midnight = time.mktime((local.tm_year, local.tm_mon, local.tm_mday, 0, 0, 0, 0, 0, -1))
        return self.s.metrics.spent_since(midnight) >= daily

    async def plan(
        self,
        api: str,
        body: dict[str, Any],
        request: Request,
    ) -> tuple[RequestView, Route]:
        view = view_request(api, body)
        key = view.session_key(request.headers)
        policy = self.s.config.policy
        session = self.s.sessions.get(
            key, now=self.s.clock(), idle_s=policy.idle_reset_minutes * 60
        )
        decision = await self.s.classifier.classify(view)
        pin = request.headers.get("x-router-model")
        if pin is None and view.requested_model in {m.name for m in self.s.config.models}:
            pin = view.requested_model
        route = select(
            self.s.config,
            decision,
            view,
            session_key=key,
            session=session,
            pin=pin,
            budget_exceeded=self._budget_exceeded(),
        )
        return view, route

    def headers(self, route: Route, model: Model, attempts: int) -> dict[str, str]:
        return {
            "x-router-model": model.name,
            "x-router-backend-model": _ascii(model.model),
            "x-router-effort": route.effort,
            "x-router-task": route.decision.task,
            "x-router-bucket": route.decision.bucket,
            "x-router-reason": _ascii(route.reason),
            "x-router-session": route.session,
            "x-router-attempts": str(attempts),
            "cache-control": "no-store",
        }

    def log_route(
        self,
        view: RequestView,
        route: Route,
        model: Model,
        status: int,
        usage: Usage,
        out_chars: int,
        started: float,
        attempts: int,
    ) -> None:
        estimated = not usage.known
        inp = usage.input_tokens if usage.known else view.context_tokens
        out = usage.output_tokens if usage.known else (out_chars + 3) // 4
        if status >= 400:
            inp = out = 0
        baseline = self.s.config.baseline()
        latency = (self.s.clock() - started) * 1000
        self.s.metrics.add(
            RouteRecord(
                session=route.session,
                api=view.api,
                task=route.decision.task,
                bucket=route.decision.bucket,
                difficulty=route.decision.difficulty,
                model=model.name,
                effort=route.effort,
                reason=route.reason[:200],
                status=status,
                input_tokens=inp,
                output_tokens=out,
                estimated=estimated,
                cost_usd=estimate_cost(model, inp, out),
                baseline_usd=estimate_cost(baseline, inp, out),
                latency_ms=latency,
                attempts=attempts,
                stream=view.stream,
                ts=self.s.clock(),
            )
        )
        log.info(
            "route api=%s task=%s bucket=%s model=%s effort=%s status=%s attempts=%s "
            "latency_ms=%.0f",
            view.api,
            route.decision.task,
            route.decision.bucket,
            model.name,
            route.effort,
            status,
            attempts,
            latency,
        )

    # ------------------------------------------------------------------ handlers
    async def chat(self, request: Request, api: str) -> Response:
        denied = self.guard(request, api)
        if denied is not None:
            return denied
        body = await self.read_json(request, api)
        if isinstance(body, Response):
            return body
        try:
            view, route = await self.plan(api, body, request)
        except NoRoute as exc:
            return JSONResponse(error_for(api, 503, str(exc)), status_code=503)
        if (
            route.reason.startswith("daily budget")
            and self.s.config.budgets.on_exceeded == "reject"
        ):
            return JSONResponse(error_for(api, 429, "daily budget reached"), status_code=429)
        return await self.forward(api, body, request, view, route)

    async def forward(
        self, api: str, body: dict[str, Any], request: Request, view: RequestView, route: Route
    ) -> Response:
        started = self.s.clock()
        last_error = "no attempt"
        last_status = 502
        policy = self.s.config.policy
        for attempt, model in enumerate(route.chain, start=1):
            backend = self.s.config.backends[model.backend]
            payload = prepare_body(api, body, model, backend, route.effort, policy)
            upstream_request = self.s.upstream.build(backend, payload, request.headers, api)
            try:
                response = await self.s.upstream.send(backend, upstream_request, view.stream)
            except httpx.HTTPError as exc:
                last_error, last_status = f"{model.name}: {type(exc).__name__}", 502
                log.warning("upstream %s failed: %s", model.name, type(exc).__name__)
                continue
            passthrough = {
                k: v for k, v in response.headers.items() if k.lower() in UPSTREAM_RESPONSE_HEADERS
            }
            if response.status_code >= 400:
                raw = await response.aread()
                await response.aclose()
                message = upstream_error_message(raw)
                if response.status_code in RETRYABLE and attempt < len(route.chain):
                    last_error, last_status = f"{model.name}: HTTP {response.status_code}", 502
                    log.warning(
                        "upstream %s returned %s, escalating", model.name, response.status_code
                    )
                    continue
                self.log_route(
                    view, route, model, response.status_code, Usage(), 0, started, attempt
                )
                return JSONResponse(
                    error_for(api, response.status_code, message),
                    status_code=response.status_code,
                    headers={**passthrough, **self.headers(route, model, attempt)},
                )
            if view.stream:
                return self.stream(
                    api, body, response, view, route, model, attempt, started, passthrough
                )
            raw = await response.aread()
            await response.aclose()
            try:
                data = json.loads(raw)
            except ValueError:
                last_error, last_status = f"{model.name}: invalid JSON", 502
                continue
            refused = _looks_refused(backend.api, data)
            if refused and policy.escalate_on_refusal and attempt < len(route.chain):
                nxt = route.chain[attempt]
                if nxt.quality > model.quality:
                    log.info("refusal from %s, escalating to %s", model.name, nxt.name)
                    last_error = f"{model.name}: refusal"
                    continue
            usage = usage_from(backend.api, data)
            if api != backend.api:
                data = (
                    openai_to_anthropic_response(data, model.model)
                    if api == "anthropic"
                    else anthropic_to_openai_response(data, model.model)
                )
            record(
                self.s.sessions,
                route.session,
                model,
                refused=refused,
                now=self.s.clock(),
                sticky_mode=policy.sticky,
            )
            self.log_route(view, route, model, 200, usage, len(raw), started, attempt)
            return JSONResponse(
                data, headers={**passthrough, **self.headers(route, model, attempt)}
            )
        self.log_route(
            view, route, route.chain[-1], last_status, Usage(), 0, started, len(route.chain)
        )
        return JSONResponse(
            error_for(api, last_status, f"all backends failed ({last_error})"),
            status_code=last_status,
            headers=self.headers(route, route.chain[-1], len(route.chain)),
        )

    def stream(
        self,
        api: str,
        body: dict[str, Any],
        response: httpx.Response,
        view: RequestView,
        route: Route,
        model: Model,
        attempt: int,
        started: float,
        passthrough: dict[str, str],
    ) -> StreamingResponse:
        backend = self.s.config.backends[model.backend]
        options = body.get("stream_options")
        include_usage = (
            api == "openai" and isinstance(options, dict) and bool(options.get("include_usage"))
        )
        adapter = make_adapter(api, backend.api, model.model, include_usage=include_usage)

        async def events() -> AsyncIterator[bytes]:
            status = 200
            try:
                async for chunk in response.aiter_raw():
                    out = adapter.feed(chunk)
                    if out:
                        yield out
                tail = adapter.finish()
                if tail:
                    yield tail
            except httpx.HTTPError:
                status = 502
                log.warning("upstream %s stream broke", model.name)
            finally:
                await response.aclose()
                refused = adapter.stop_reason == "refusal"
                if adapter.error:
                    status = 502
                record(
                    self.s.sessions,
                    route.session,
                    model,
                    refused=refused,
                    now=self.s.clock(),
                    sticky_mode=self.s.config.policy.sticky,
                )
                self.log_route(
                    view, route, model, status, adapter.usage, adapter.text_chars, started, attempt
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={**passthrough, **self.headers(route, model, attempt)},
        )

    async def count_tokens(self, request: Request) -> Response:
        denied = self.guard(request, "anthropic")
        if denied is not None:
            return denied
        body = await self.read_json(request, "anthropic")
        if isinstance(body, Response):
            return body
        view = view_request("anthropic", body)
        return JSONResponse(
            {"input_tokens": view.context_tokens}, headers={"x-router-estimate": "true"}
        )

    async def models(self, request: Request) -> Response:
        denied = self.guard(request, "openai")
        if denied is not None:
            return denied
        created = int(self.s.clock())
        items = [
            {
                "id": "auto",
                "object": "model",
                "type": "model",
                "created": created,
                "owned_by": "agent-model-router",
                "display_name": "auto (routed)",
            }
        ]
        items += [
            {
                "id": m.name,
                "object": "model",
                "type": "model",
                "created": created,
                "owned_by": m.backend,
                "display_name": f"{m.name} ({m.model})",
            }
            for m in self.s.config.usable_models()
        ]
        return JSONResponse({"object": "list", "data": items, "has_more": False})

    async def explain(self, request: Request) -> Response:
        api = request.query_params.get("api", "anthropic")
        if api not in {"anthropic", "openai"}:
            return JSONResponse(
                error_for("openai", 400, "api must be anthropic or openai"), status_code=400
            )
        denied = self.guard(request, api)
        if denied is not None:
            return denied
        body = await self.read_json(request, api)
        if isinstance(body, Response):
            return body
        try:
            view, route = await self.plan(api, body, request)
        except NoRoute as exc:
            return JSONResponse(error_for(api, 503, str(exc)), status_code=503)
        result = route.as_dict()
        result["baseline_model"] = self.s.config.baseline().name
        result["context_tokens"] = view.context_tokens
        result["tool_followup"] = view.tool_followup
        return JSONResponse(result, headers=self.headers(route, route.model, 0))

    async def stats(self, request: Request) -> Response:
        denied = self.guard(request, "openai")
        if denied is not None:
            return denied
        since = float(request.query_params.get("since", "0") or 0)
        summary = self.s.metrics.summary(since)
        summary["baseline_model"] = self.s.config.baseline().name
        return JSONResponse(summary)

    async def health(self, request: Request) -> Response:
        denied = self.guard(request, "openai", auth=False)
        if denied is not None:
            return denied
        return JSONResponse({"status": "ok", "version": __version__})


def create_app(
    config: Config,
    *,
    token: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    metrics: MetricsStore | None = None,
    clock: Callable[[], float] = time.time,
) -> Starlette:
    """Build the ASGI app. ``token=None`` disables auth (only valid with ``server.auth: false``)."""
    upstream = Upstream(config, transport=transport)
    state = RouterState(
        config=config,
        upstream=upstream,
        classifier=_build_classifier(config, upstream),
        sessions=SessionStore(),
        metrics=metrics if metrics is not None else MetricsStore(config.metrics_db),
        token=token,
        clock=clock,
    )
    if config.server.auth and token is None:
        raise ValueError("server.auth is on: pass the router token")
    router = Router(state)

    async def messages(request: Request) -> Response:
        return await router.chat(request, "anthropic")

    async def completions(request: Request) -> Response:
        return await router.chat(request, "openai")

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        yield
        await upstream.aclose()

    app = Starlette(
        routes=[
            StarletteRoute("/v1/messages", messages, methods=["POST"]),
            StarletteRoute("/v1/messages/count_tokens", router.count_tokens, methods=["POST"]),
            StarletteRoute("/v1/chat/completions", completions, methods=["POST"]),
            StarletteRoute("/chat/completions", completions, methods=["POST"]),
            StarletteRoute("/v1/models", router.models, methods=["GET"]),
            StarletteRoute("/router/explain", router.explain, methods=["POST"]),
            StarletteRoute("/router/stats", router.stats, methods=["GET"]),
            StarletteRoute("/healthz", router.health, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
    app.state.router = state
    return app
