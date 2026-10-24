"""The bundled synthetic training set: templated coding-agent prompts with labels.

Generated deterministically (``random.Random(seed)``), so ``agent-model-router train``
reproduces the shipped weights bit for bit and anyone can inspect what the scorer learned
from. Labels: ``task`` (chat/edit/plan/debug/review/tool_followup) and ``level`` 0..4
(trivial..expert). It is synthetic: tune the quality bars on your own prompts too.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from importlib import resources
from pathlib import Path
from typing import Any

from agent_model_router.view import RequestView

__all__ = ["generate", "load_rows", "row_view", "write_dataset"]

_SLOTS: dict[str, tuple[str, ...]] = {
    "file": (
        "app/models.py",
        "src/server.ts",
        "cmd/main.go",
        "lib/parser.rs",
        "api/views.py",
        "web/src/App.tsx",
        "pkg/cache/lru.go",
        "services/billing.py",
        "README.md",
        "src/utils/date.js",
        "tests/test_api.py",
        "config/settings.yaml",
    ),
    "var": ("user_id", "maxRetries", "tmp", "data", "cfg", "handler", "result", "idx", "conn"),
    "var2": ("account_id", "retryLimit", "buffer", "payload", "settings", "on_event", "output"),
    "func": (
        "parse_date",
        "fetchUser",
        "build_index",
        "render",
        "validate_email",
        "retry",
        "compute_total",
        "merge_configs",
        "serialize",
    ),
    "thing": ("orders", "users", "invoices", "comments", "search", "uploads", "sessions"),
    "lib": ("requests", "lodash", "pydantic", "express", "sqlalchemy", "react-query", "tokio"),
    "module": (
        "auth",
        "billing",
        "ingestion pipeline",
        "scheduler",
        "storage layer",
        "router",
        "plugin system",
        "event bus",
    ),
    "system": (
        "rate limiter",
        "job queue",
        "feature-flag service",
        "cache",
        "search index",
        "notification service",
        "metrics pipeline",
    ),
}

# (task, level, template). Levels: 0 trivial, 1 easy, 2 medium, 3 hard, 4 expert.
_TEMPLATES: tuple[tuple[str, int, str], ...] = (
    ("chat", 0, "thanks!"),
    ("chat", 0, "ok, looks good"),
    ("chat", 0, "yes"),
    ("chat", 0, "What does {func} return?"),
    ("chat", 0, "What is the default port for {lib}?"),
    ("chat", 1, "Explain what {func} in {file} does."),
    ("chat", 1, "What's the difference between a list and a tuple in Python?"),
    ("chat", 1, "How do I install {lib}?"),
    ("chat", 2, "Explain how the {module} works and how data flows through it."),
    ("chat", 2, "Why would someone use {lib} instead of writing it by hand? Compare both."),
    (
        "chat",
        3,
        "Explain the concurrency model of the {module} and where race conditions could appear.",
    ),
    ("edit", 0, "Rename {var} to {var2} in {file}."),
    ("edit", 0, "Fix the typo in the docstring of {func}."),
    ("edit", 0, "Sort imports in {file}."),
    ("edit", 0, "Remove the unused import in {file}."),
    ("edit", 0, "Bump {lib} to the latest version."),
    ("edit", 0, "Add a comment above {func} explaining the return value."),
    ("edit", 1, "Add a docstring and type hints to {func} in {file}."),
    ("edit", 1, "Add a unit test for {func} covering the empty input case."),
    ("edit", 1, "Make {func} return None instead of raising when the input is empty."),
    ("edit", 1, "Add a --verbose flag to the CLI in {file}."),
    ("edit", 2, "Add pagination to the {thing} endpoint in {file} and update the tests."),
    ("edit", 2, "Implement caching for {func} with a 5 minute TTL and add tests."),
    ("edit", 2, "Add input validation to the {thing} form and show errors inline in {file}."),
    ("edit", 2, "Write a migration script that adds a status column to {thing} and backfills it."),
    ("edit", 2, "Convert {file} from callbacks to async/await and keep the public API."),
    (
        "edit",
        3,
        "Refactor the {module} to use dependency injection across the codebase, update every module that constructs it, and keep all tests green.",
    ),
    (
        "edit",
        3,
        "Rewrite the {module} so it is thread-safe and add tests that prove there is no race condition.",
    ),
    ("edit", 3, "Migrate the {module} from {lib} to the standard library across the entire repo."),
    (
        "edit",
        3,
        "Implement optimistic concurrency for {thing} updates end-to-end: API, storage and client retries.",
    ),
    (
        "edit",
        4,
        "Rewrite the {system} as a distributed, multi-tenant service with sharding and replication; keep the existing protocol compatible and write a migration plan for production data.",
    ),
    (
        "edit",
        4,
        "Implement a lock-free version of the {system} with a formal argument for its invariants and a performance benchmark.",
    ),
    ("plan", 1, "Outline the steps to add a {thing} export button."),
    ("plan", 2, "Plan how we should add {thing} search. List options and trade-offs."),
    ("plan", 2, "What approach would you take to split {file} into smaller modules?"),
    (
        "plan",
        3,
        "Design the architecture for a {system} that handles 10k requests per second. Compare approaches and trade-offs.",
    ),
    (
        "plan",
        3,
        "Write an RFC for migrating the {module} to an event-driven architecture, with milestones.",
    ),
    (
        "plan",
        4,
        "Design a distributed {system} with consensus, failure handling, cache invalidation and a migration strategy from the current monolith. Cover trade-offs, risks and a roadmap.",
    ),
    (
        "plan",
        4,
        "Propose an architecture for multi-region active-active {thing} storage with conflict resolution; cover consistency, latency and the migration plan.",
    ),
    ("debug", 1, "{func} returns the wrong value for an empty list, can you fix it?"),
    ("debug", 1, "The test test_{func} is failing with an AssertionError. Fix it."),
    (
        "debug",
        2,
        'Why does {func} crash with a KeyError when {var} is missing?\n\nTraceback (most recent call last):\n  File "{file}", line 42, in {func}\n    return data["{var}"]\nKeyError: \'{var}\'',
    ),
    (
        "debug",
        2,
        "The {thing} endpoint fails with a 500 after the last deploy. Here is the error:\nTypeError: Cannot read properties of undefined (reading '{var}')\n    at {func} ({file}:88:12)",
    ),
    (
        "debug",
        3,
        "There's a flaky test in the {module}; it fails about one run in twenty with a timeout. I suspect a race condition or deadlock in the async code. Find the root cause.",
    ),
    (
        "debug",
        3,
        "Memory usage of the {system} grows until it crashes after a few hours. Find the memory leak and fix it.",
    ),
    (
        "debug",
        4,
        "Under load the {system} loses writes across replicas about once a day. We suspect the consensus or replication protocol. Find the bug, explain the invariant that breaks and fix it.",
    ),
    (
        "review",
        1,
        "Quick review of this small diff, anything obviously wrong?\n```diff\n- {var} = 1\n+ {var2} = 1\n```",
    ),
    (
        "review",
        2,
        "Review the pull request for the {thing} feature and give feedback on naming, tests and error handling.",
    ),
    (
        "review",
        3,
        "Do a security review of the {module}: auth, permission checks, secrets handling and injection risks.",
    ),
    (
        "review",
        3,
        "Audit the {module} for performance problems and concurrency bugs and suggest fixes.",
    ),
    (
        "review",
        4,
        "Audit the whole repo's architecture: module boundaries, the {system} protocol, failure modes and security; produce a prioritised report.",
    ),
)

_PREAMBLES = ("", "", "", "Hey, ", "Please ", "Can you ", "Quick one: ")
_CODE_SNIPPET = "```python\ndef {func}(items):\n    total = 0\n    for item in items:\n        total += item.price\n    return total\n```"


def _fill(template: str, rnd: random.Random) -> str:
    out = template
    for slot, values in _SLOTS.items():
        token = "{" + slot + "}"
        while token in out:
            out = out.replace(token, rnd.choice(values), 1)
    return out


def generate(n: int = 1500, seed: int = 7) -> Iterator[dict[str, Any]]:
    """Yield ``n`` labelled rows."""
    # Seeded PRNG for reproducible data, not for anything security related.
    rnd = random.Random(seed)
    for i in range(n):
        task, level, template = rnd.choice(_TEMPLATES)
        text = _fill(template, rnd)
        # A tool-loop step: the last message is a tool result; the human text is the one
        # that started the loop, and the difficulty stays that of the loop.
        tool_followup = rnd.random() < 0.08
        rows_task = "tool_followup" if tool_followup else task
        if text and text[0].isupper() and rnd.random() < 0.3:
            pre = rnd.choice(_PREAMBLES)
            text = pre + (text[0].lower() + text[1:] if pre else text)
        if task == "edit" and level >= 1 and rnd.random() < 0.25:
            text += "\n\n" + _fill(_CODE_SNIPPET, rnd)
        tools = rnd.random() < 0.85
        turns = max(1, int(rnd.expovariate(1 / (3 + level))))
        context = int(rnd.lognormvariate(8.0 + 0.15 * level, 1.2))
        yield {
            "id": i,
            "task": rows_task,
            "level": level,
            "text": text,
            "tools": tools,
            "tool_followup": tool_followup,
            "turns": turns,
            "context_tokens": context,
        }


def row_view(row: dict[str, Any]) -> RequestView:
    """The :class:`RequestView` a row stands for."""
    return RequestView(
        api="anthropic",
        last_user_text=str(row["text"]),
        earlier_user_texts=("earlier",) * (int(row.get("turns", 1)) - 1),
        last_assistant_text="",
        system_text="",
        tools_present=bool(row.get("tools")),
        tool_followup=bool(row.get("tool_followup")),
        user_turns=int(row.get("turns", 1)),
        recent_tool_results=1 if row.get("tool_followup") else 0,
        context_tokens=int(row.get("context_tokens", 0)),
        max_tokens=None,
        stream=False,
        requested_model=None,
    )


def write_dataset(path: Path, n: int = 1500, seed: int = 7) -> int:
    rows = list(generate(n, seed))
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), "utf-8")
    return len(rows)


def load_rows(
    name: str = "synthetic.jsonl", path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Rows from a bundled JSONL file (``synthetic.jsonl`` or ``eval.jsonl``) or a path."""
    if path is not None:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    else:
        text = resources.files("agent_model_router").joinpath("data", name).read_text("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
