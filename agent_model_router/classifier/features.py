"""Feature extraction: a request view becomes a small vector of named numbers in [0, 1].

The features are deliberately few and readable, so every routing decision can be
explained as "these features pushed the score up or down by this much".
"""

from __future__ import annotations

import math
import re

from agent_model_router.view import RequestView

__all__ = ["FEATURE_NAMES", "KEYWORDS", "extract", "keyword_hits"]

KEYWORDS: dict[str, tuple[str, ...]] = {
    "trivial": (
        "rename",
        "typo",
        "spelling",
        "format",
        "reformat",
        "lint",
        "indent",
        "comment",
        "docstring",
        "bump",
        "one-line",
        "one line",
        "quick",
        "simple",
        "small",
        "tiny",
        "trailing",
        "whitespace",
        "sort imports",
        "unused import",
        "print",
        "log line",
        "what is",
        "what does",
        "thanks",
        "thank you",
        "ok",
        "yes",
        "looks good",
    ),
    "hard": (
        "architecture",
        "architect",
        "design",
        "redesign",
        "refactor",
        "migrate",
        "migration",
        "concurrency",
        "concurrent",
        "race condition",
        "deadlock",
        "thread-safe",
        "async",
        "performance",
        "optimize",
        "optimise",
        "latency",
        "memory leak",
        "scalab",
        "distributed",
        "consensus",
        "algorithm",
        "complexity",
        "across the codebase",
        "entire",
        "whole repo",
        "every module",
        "end-to-end",
        "protocol",
        "invariant",
        "proof",
        "formal",
        "rewrite",
        "framework",
        "multi-tenant",
        "sharding",
        "replication",
        "cache invalidation",
    ),
    "debug": (
        "error",
        "traceback",
        "exception",
        "fails",
        "failing",
        "failed",
        "bug",
        "crash",
        "stack trace",
        "segfault",
        "broken",
        "doesn't work",
        "does not work",
        "why does",
        "flaky",
        "regression",
        "panic",
        "undefined",
        "nullpointer",
        "keyerror",
        "typeerror",
        "assertionerror",
        "leak",
        "oom",
        "root cause",
        "find the bug",
        "times out",
    ),
    "plan": (
        "plan",
        "approach",
        "strategy",
        "proposal",
        "rfc",
        "trade-off",
        "tradeoff",
        "roadmap",
        "options",
        "should we",
        "how would you",
        "outline",
        "milestones",
        "spec",
    ),
    "review": (
        "review",
        "pull request",
        "pr",
        "diff",
        "feedback",
        "audit",
        "critique",
        "code smell",
        "look over",
        "sanity check",
    ),
    "edit": (
        "add",
        "implement",
        "write",
        "create",
        "change",
        "update",
        "fix",
        "remove",
        "delete",
        "replace",
        "convert",
        "extract",
        "move",
        "make",
        "support",
        "wire",
        "hook up",
        "rename",
        "refactor",
        "rewrite",
        "migrate",
        "bump",
        "turn",
    ),
    "scope": (
        "and update",
        "and add",
        "and keep",
        "and show",
        "tests",
        "endpoint",
        "pagination",
        "validation",
        "cach",
        "retries",
        "backoff",
        "export",
        "integrat",
        "feature",
        "both",
        "backend",
        "frontend",
        "api",
        "schema",
        "migration script",
        "flag",
        "ttl",
        "async/await",
    ),
    "sensitive": (
        "production",
        "prod",
        "credential",
        "secret",
        "password",
        "billing",
        "payment",
        "permission",
        "auth",
        "security",
        "vulnerab",
        "drop table",
        "rm -rf",
        "force push",
        "migration",
        "encryption",
        "private key",
        "gdpr",
        "pii",
    ),
}

FEATURE_NAMES = (
    "chars",
    "lines",
    "context",
    "code_blocks",
    "files",
    "tools",
    "tool_followup",
    "turns",
    "kw_trivial",
    "kw_hard",
    "kw_debug",
    "kw_plan",
    "kw_review",
    "kw_scope",
    "kw_sensitive",
    "traceback",
    "short_question",
    "steps",
)

_CODE_BLOCK = re.compile(r"```")
_FILE = re.compile(
    r"(?:[\w.\-]+/)*[\w\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|rb|php|cs|c|cc|"
    r"cpp|h|hpp|sql|yml|yaml|toml|json|md|sh|tf|swift|scala|vue|svelte|css|html)\b"
)
_TRACEBACK = re.compile(
    r"Traceback \(most recent call last\)|^\s+at [\w.$<>]+\(.*:\d+\)|"
    r"\b\w+(Error|Exception): |panicked at|FAILED |error\[E\d+\]",
    re.M,
)
_STEPS = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+\S", re.M)


def _pattern(keyword: str) -> re.Pattern[str]:
    # Keywords match at a word start; short ones (and "pr", "ok") must be whole words.
    tail = r"(?![a-z])" if len(keyword) <= 4 else ""
    return re.compile(r"(?<![a-z])" + re.escape(keyword) + tail)


_PATTERNS = {group: [_pattern(kw) for kw in kws] for group, kws in KEYWORDS.items()}


def keyword_hits(text: str, group: str) -> int:
    """How many distinct keywords of ``group`` occur in ``text``."""
    lowered = text.lower()
    return sum(1 for pattern in _PATTERNS[group] if pattern.search(lowered))


def _log_scale(value: float, cap: float) -> float:
    return min(1.0, math.log1p(max(0.0, value)) / math.log1p(cap))


def extract(view: RequestView) -> dict[str, float]:
    """Features of the latest human message, in [0, 1]."""
    text = view.last_user_text
    # Code inside fences counts for size but not for keywords: a pasted file full of
    # "async" should not by itself make a request hard.
    prose = re.sub(r"```.*?```", " ", text, flags=re.S)
    lines = text.count("\n") + 1 if text else 0
    stripped = text.strip()
    return {
        "chars": _log_scale(len(text), 20000),
        "lines": _log_scale(lines, 400),
        "context": _log_scale(view.context_tokens, 200000),
        "code_blocks": min(1.0, len(_CODE_BLOCK.findall(text)) / 2 / 4),
        "files": min(1.0, len(set(_FILE.findall(text))) / 6),
        "tools": 1.0 if view.tools_present else 0.0,
        "tool_followup": 1.0 if view.tool_followup else 0.0,
        "turns": _log_scale(view.user_turns, 50),
        "kw_trivial": min(1.0, keyword_hits(prose, "trivial") / 2),
        "kw_hard": min(1.0, keyword_hits(prose, "hard") / 3),
        "kw_debug": min(1.0, keyword_hits(prose, "debug") / 2),
        "kw_plan": min(1.0, keyword_hits(prose, "plan") / 2),
        "kw_review": min(1.0, keyword_hits(prose, "review") / 2),
        "kw_scope": min(1.0, keyword_hits(prose, "scope") / 3),
        "kw_sensitive": min(1.0, keyword_hits(prose, "sensitive") / 2),
        "traceback": 1.0 if _TRACEBACK.search(text) else 0.0,
        "short_question": 1.0 if stripped.endswith("?") and len(stripped) < 160 else 0.0,
        "steps": min(1.0, len(_STEPS.findall(prose)) / 6),
    }
