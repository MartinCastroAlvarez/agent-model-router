"""Train and evaluate the difficulty scorer on labelled rows (bundled or your own JSONL)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_model_router.classifier.dataset import row_view
from agent_model_router.classifier.features import extract
from agent_model_router.classifier.heuristic import HeuristicClassifier
from agent_model_router.classifier.heuristic import bucket_index
from agent_model_router.classifier.linear import LinearScorer
from agent_model_router.classifier.linear import train

__all__ = ["evaluate", "train_rows"]


def train_rows(rows: Sequence[dict[str, Any]]) -> LinearScorer:
    """Fit the scorer: target = level / 4."""
    data = [(extract(row_view(r)), int(r["level"]) / 4) for r in rows]
    return train(data)


def evaluate(scorer: LinearScorer, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Bucket accuracy (exact and within one), task accuracy, mean absolute level error."""
    clf = HeuristicClassifier(scorer)
    exact = within = task_ok = 0
    abs_err = 0.0
    for r in rows:
        d = clf.decide(row_view(r))
        got = bucket_index(d.bucket)
        want = int(r["level"])
        exact += got == want
        within += abs(got - want) <= 1
        abs_err += abs(got - want)
        task_ok += d.task == r["task"]
    n = max(1, len(rows))
    return {
        "rows": float(len(rows)),
        "bucket_exact": round(exact / n, 4),
        "bucket_within_one": round(within / n, 4),
        "task_accuracy": round(task_ok / n, 4),
        "mean_abs_level_error": round(abs_err / n, 4),
    }
