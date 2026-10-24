"""Routing policy: from a decision and the pool to a model, an effort and a fallback chain.

Rule: the cheapest model whose quality clears the bar for the request's difficulty bucket,
among the models that fit (context window, tool support, budget, secrets stay local).
Within a conversation the model only moves up (``ratchet``), which keeps the provider's
prompt cache warm; tool-loop steps keep the model of the human turn that started them.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field

from agent_model_router.classifier.heuristic import Decision
from agent_model_router.config import Config
from agent_model_router.config import Model
from agent_model_router.view import RequestView

__all__ = [
    "NoRoute",
    "Route",
    "SessionState",
    "SessionStore",
    "estimate_cost",
    "record",
    "select",
]

_EXPECTED_OUTPUT = 1500


class NoRoute(Exception):
    """No model in the pool can take this request."""


@dataclass
class SessionState:
    model: str
    quality: float
    last_seen: float
    escalate_next: bool = False


class SessionStore:
    """In-memory, bounded LRU of conversation -> model. Keys are hashes, never content."""

    def __init__(self, max_entries: int = 10000) -> None:
        self.max_entries = max_entries
        self._data: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, *, now: float, idle_s: float) -> SessionState | None:
        with self._lock:
            state = self._data.get(key)
            if state is None:
                return None
            if now - state.last_seen > idle_s:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return state

    def put(self, key: str, state: SessionState) -> None:
        with self._lock:
            self._data[key] = state
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


@dataclass(frozen=True)
class Route:
    model: Model
    effort: str
    reason: str
    chain: tuple[Model, ...]
    decision: Decision
    session: str
    sticky: bool = False
    candidates: tuple[dict[str, object], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model.name,
            "backend_model": self.model.model,
            "backend": self.model.backend,
            "effort": self.effort,
            "reason": self.reason,
            "sticky": self.sticky,
            "fallback_chain": [m.name for m in self.chain],
            "session": self.session,
            "candidates": list(self.candidates),
            "decision": self.decision.as_dict(),
        }


def estimate_cost(model: Model, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * model.price_in + output_tokens * model.price_out) / 1_000_000


def _is_trusted(config: Config, model: Model) -> bool:
    return config.backends[model.backend].trusted


def select(
    config: Config,
    decision: Decision,
    view: RequestView,
    *,
    session_key: str,
    session: SessionState | None,
    pin: str | None = None,
    budget_exceeded: bool = False,
) -> Route:
    policy = config.policy
    expected_out = min(view.max_tokens or _EXPECTED_OUTPUT, _EXPECTED_OUTPUT)
    needed_ctx = view.context_tokens + min(view.max_tokens or 1024, 8192)
    usable = config.usable_models()
    if not usable:
        raise NoRoute("no usable model: enable a backend (and export its key)")

    notes: list[str] = []
    rows: list[dict[str, object]] = []
    fit: list[Model] = []
    bar = policy.quality[decision.bucket]
    for m in usable:
        why = None
        if view.tools_present and not m.tools:
            why = "no tool support"
        elif needed_ctx > m.context:
            why = f"context {needed_ctx} > {m.context}"
        est = estimate_cost(m, view.context_tokens, expected_out)
        rows.append(
            {
                "model": m.name,
                "quality": m.quality,
                "clears_bar": m.quality >= bar,
                "est_cost_usd": round(est, 6),
                "excluded": why,
            }
        )
        if why is None:
            fit.append(m)
    if not fit:
        raise NoRoute("no model fits this request (context window / tools)")

    if decision.secret:
        local = [m for m in fit if _is_trusted(config, m)]
        if local:
            fit = local
            notes.append("secret detected: trusted (local) backends only")

    def cost(m: Model) -> float:
        return estimate_cost(m, view.context_tokens, expected_out)

    chosen: Model
    if budget_exceeded:
        chosen = min(fit, key=lambda m: (cost(m), -m.quality))
        reason = "daily budget reached: cheapest model"
    elif pin and pin in {m.name for m in fit}:
        chosen = next(m for m in fit if m.name == pin)
        reason = f"pinned by client ({pin})"
    else:
        clearing = [m for m in fit if m.quality >= bar]
        if clearing:
            chosen = min(clearing, key=lambda m: (cost(m), -m.quality))
            reason = f"{decision.bucket} {decision.task}: cheapest model with quality >= {bar}"
        else:
            chosen = max(fit, key=lambda m: (m.quality, -cost(m)))
            reason = f"{decision.bucket} {decision.task}: no model clears {bar}, using the best"

    sticky = False
    if session is not None and policy.sticky == "ratchet" and not budget_exceeded and not pin:
        previous = next((m for m in fit if m.name == session.model), None)
        if previous is not None:
            if session.escalate_next:
                higher = sorted(
                    (m for m in fit if m.quality > previous.quality),
                    key=lambda m: (m.quality, cost(m)),
                )
                if higher and higher[0].quality > chosen.quality:
                    chosen = higher[0]
                    reason = "escalated after a refusal in this conversation"
                elif previous.quality >= chosen.quality:
                    chosen, sticky = previous, True
                    reason = "sticky (no higher model to escalate to)"
            elif view.tool_followup:
                chosen, sticky = previous, True
                reason = "tool loop keeps the conversation's model"
            elif previous.quality >= chosen.quality:
                if previous.name != chosen.name:
                    reason = f"sticky: conversation already on {previous.name} (ratchet)"
                    sticky = True
                chosen = previous
            else:
                reason += f" (up from {previous.name})"

    up = sorted((m for m in fit if m.quality > chosen.quality), key=lambda m: (m.quality, cost(m)))
    down = sorted(
        (m for m in fit if m.quality <= chosen.quality and m is not chosen),
        key=lambda m: (-m.quality, cost(m)),
    )
    chain = tuple([chosen, *up, *down][: policy.max_attempts])
    if notes:
        reason = "; ".join([reason, *notes])
    return Route(
        model=chosen,
        effort=policy.effort[decision.bucket],
        reason=reason,
        chain=chain,
        decision=decision,
        session=session_key,
        sticky=sticky,
        candidates=tuple(rows),
    )


def record(
    store: SessionStore,
    key: str,
    model: Model,
    *,
    refused: bool = False,
    now: float | None = None,
    sticky_mode: str = "ratchet",
) -> None:
    """Remember the conversation's model after a request (ratchet: never downgrade)."""
    if sticky_mode == "off":
        return
    now = time.time() if now is None else now
    store.put(
        key,
        SessionState(model=model.name, quality=model.quality, last_seen=now, escalate_next=refused),
    )
