"""Local SQLite route log and the ``stats`` report.

One row per request: when, which conversation hash, task, bucket, model, effort, status,
token counts, estimated cost and what the baseline model would have cost. Never prompt
text, never headers or keys. All SQL is parameterised.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["MetricsStore", "RouteRecord", "format_stats"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session TEXT NOT NULL,
    api TEXT NOT NULL,
    task TEXT NOT NULL,
    bucket TEXT NOT NULL,
    difficulty REAL NOT NULL,
    model TEXT NOT NULL,
    effort TEXT NOT NULL,
    reason TEXT NOT NULL,
    status INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    estimated INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    baseline_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    attempts INTEGER NOT NULL,
    stream INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS routes_ts ON routes (ts);
"""

_COLUMNS = (
    "ts",
    "session",
    "api",
    "task",
    "bucket",
    "difficulty",
    "model",
    "effort",
    "reason",
    "status",
    "input_tokens",
    "output_tokens",
    "estimated",
    "cost_usd",
    "baseline_usd",
    "latency_ms",
    "attempts",
    "stream",
)
_INSERT = (
    "INSERT INTO routes (ts, session, api, task, bucket, difficulty, model, effort, reason, "
    "status, input_tokens, output_tokens, estimated, cost_usd, baseline_usd, latency_ms, "
    "attempts, stream) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


@dataclass(frozen=True)
class RouteRecord:
    session: str
    api: str
    task: str
    bucket: str
    difficulty: float
    model: str
    effort: str
    reason: str
    status: int
    input_tokens: int
    output_tokens: int
    estimated: bool
    cost_usd: float
    baseline_usd: float
    latency_ms: float
    attempts: int
    stream: bool
    ts: float = 0.0


class MetricsStore:
    def __init__(self, path: str | Path | None) -> None:
        """``path=None`` keeps the log in memory (tests, ``metrics.enabled: false``)."""
        self._lock = threading.Lock()
        if path is None:
            target = ":memory:"
        else:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target = str(p)
        self.conn = sqlite3.connect(target, check_same_thread=False)
        # Close the connection when the store is collected, even if close() is never called.
        self._finalizer = weakref.finalize(self, self.conn.close)
        self.conn.executescript(_SCHEMA)
        if path is not None:
            Path(target).chmod(0o600)

    def add(self, record: RouteRecord) -> None:
        values = [getattr(record, c) for c in _COLUMNS]
        if not values[0]:
            values[0] = time.time()
        values = [int(v) if isinstance(v, bool) else v for v in values]
        with self._lock:
            self.conn.execute(_INSERT, values)
            self.conn.commit()

    def spent_since(self, since: float) -> float:
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM routes WHERE ts >= ?", (since,)
            ).fetchone()
        return float(row[0])

    def summary(self, since: float = 0.0) -> dict[str, Any]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT model, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd), "
                "SUM(baseline_usd), AVG(latency_ms), SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 "
                "END), SUM(CASE WHEN attempts > 1 THEN 1 ELSE 0 END) FROM routes WHERE ts >= ? "
                "GROUP BY model ORDER BY COUNT(*) DESC",
                (since,),
            ).fetchall()
            buckets = self.conn.execute(
                "SELECT bucket, model, COUNT(*) FROM routes WHERE ts >= ? GROUP BY bucket, model",
                (since,),
            ).fetchall()
            tasks = self.conn.execute(
                "SELECT task, COUNT(*) FROM routes WHERE ts >= ? GROUP BY task ORDER BY 2 DESC",
                (since,),
            ).fetchall()
        models = [
            {
                "model": r[0],
                "requests": r[1],
                "input_tokens": r[2] or 0,
                "output_tokens": r[3] or 0,
                "cost_usd": r[4] or 0.0,
                "baseline_usd": r[5] or 0.0,
                "avg_latency_ms": round(r[6] or 0.0, 1),
                "errors": r[7] or 0,
                "escalated": r[8] or 0,
            }
            for r in rows
        ]
        cost = sum(m["cost_usd"] for m in models)
        baseline = sum(m["baseline_usd"] for m in models)
        return {
            "requests": sum(m["requests"] for m in models),
            "cost_usd": cost,
            "baseline_usd": baseline,
            "savings_usd": baseline - cost,
            "savings_pct": (100.0 * (baseline - cost) / baseline) if baseline else 0.0,
            "models": models,
            "buckets": [{"bucket": b, "model": m, "requests": n} for b, m, n in buckets],
            "tasks": [{"task": t, "requests": n} for t, n in tasks],
        }

    def close(self) -> None:
        with self._lock:
            self._finalizer()


def format_stats(summary: dict[str, Any], baseline_name: str) -> str:
    lines = [
        f"{'model':<16}{'requests':>9}{'in tok':>11}{'out tok':>10}{'cost $':>11}"
        f"{'errors':>8}{'escalated':>11}{'avg ms':>9}"
    ]
    for m in summary["models"]:
        lines.append(
            f"{m['model']:<16}{m['requests']:>9}{m['input_tokens']:>11}{m['output_tokens']:>10}"
            f"{m['cost_usd']:>11.6f}{m['errors']:>8}{m['escalated']:>11}{m['avg_latency_ms']:>9}"
        )
    lines.append("")
    lines.append(f"requests: {summary['requests']}")
    rows = [
        ("routed cost:", summary["cost_usd"]),
        (f"always-{baseline_name}:", summary["baseline_usd"]),
    ]
    width = max(len(label) for label, _ in rows) + 1
    for label, value in rows:
        lines.append(f"{label:<{width}}${value:.6f}")
    pct = summary["savings_pct"]
    lines.append(f"{'saved:':<{width}}${summary['savings_usd']:.6f} ({pct:.1f}%)")
    if summary["tasks"]:
        lines.append(
            "tasks: " + ", ".join(f"{t['task']}={t['requests']}" for t in summary["tasks"])
        )
    return "\n".join(lines)
