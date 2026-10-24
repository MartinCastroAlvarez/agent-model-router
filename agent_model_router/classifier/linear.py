"""A tiny logistic-regression scorer: difficulty = sigmoid(w . x + b).

Pure Python, no numpy, deterministic. Trained with soft labels (difficulty level / 4) so
one model yields an ordinal score in [0, 1] that the policy cuts into buckets. Weights are
plain JSON (never pickle), shipped with the package and retrainable with
``agent-model-router train``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from agent_model_router.classifier.features import FEATURE_NAMES

__all__ = ["LinearScorer", "train"]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class LinearScorer:
    weights: dict[str, float]
    bias: float
    meta: dict[str, Any]

    @property
    def thresholds(self) -> tuple[float, float, float, float]:
        """Cut points between the five buckets, fitted on the training set."""
        cuts = self.meta.get("thresholds") or (0.2, 0.4, 0.6, 0.8)
        a, b, c, d = (float(x) for x in cuts)
        return (a, b, c, d)

    def score(self, features: Mapping[str, float]) -> float:
        z = self.bias + sum(self.weights.get(k, 0.0) * v for k, v in features.items())
        return _sigmoid(z)

    def contributions(self, features: Mapping[str, float]) -> list[tuple[str, float]]:
        """Per-feature contributions to the logit, largest magnitude first (non-zero only)."""
        items = [(k, self.weights.get(k, 0.0) * v) for k, v in features.items()]
        return sorted((i for i in items if abs(i[1]) > 1e-9), key=lambda i: -abs(i[1]))

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "features": list(FEATURE_NAMES),
                    "weights": self.weights,
                    "bias": self.bias,
                    "meta": self.meta,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> LinearScorer:
        data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(data.get("weights"), dict):
            raise ValueError("weights file must be a JSON object with 'weights'")
        weights = {str(k): float(v) for k, v in data["weights"].items()}
        unknown = set(weights) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown features in weights: {sorted(unknown)}")
        return cls(
            weights=weights, bias=float(data.get("bias", 0.0)), meta=dict(data.get("meta") or {})
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> LinearScorer:
        if path is None:
            text = (
                resources.files("agent_model_router")
                .joinpath("data/weights.json")
                .read_text(encoding="utf-8")
            )
        else:
            text = Path(path).expanduser().read_text(encoding="utf-8")
        return cls.from_json(text)


def train(
    rows: Sequence[tuple[Mapping[str, float], float]],
    *,
    epochs: int = 1500,
    lr: float = 2.0,
    l2: float = 1e-3,
) -> LinearScorer:
    """Full-batch gradient descent on cross-entropy with soft targets in [0, 1]."""
    if not rows:
        raise ValueError("no training rows")
    names = list(FEATURE_NAMES)
    w = [0.0] * len(names)
    b = 0.0
    n = len(rows)
    xs = [[float(x.get(k, 0.0)) for k in names] for x, _ in rows]
    ys = [float(y) for _, y in rows]
    for _ in range(epochs):
        gw = [0.0] * len(names)
        gb = 0.0
        for x, y in zip(xs, ys, strict=True):
            p = _sigmoid(b + sum(wi * xi for wi, xi in zip(w, x, strict=True)))
            err = p - y
            gb += err
            for i, xi in enumerate(x):
                if xi:
                    gw[i] += err * xi
        b -= lr * gb / n
        for i in range(len(w)):
            w[i] -= lr * (gw[i] / n + l2 * w[i])
    weights = {k: round(v, 6) for k, v in zip(names, w, strict=True)}
    bias = round(b, 6)
    scorer = LinearScorer(weights=weights, bias=bias, meta={})
    scores = [scorer.score(x) for x, _ in rows]
    levels = [round(y * 4) for y in ys]
    return LinearScorer(
        weights=weights,
        bias=bias,
        meta={
            "epochs": epochs,
            "lr": lr,
            "l2": l2,
            "rows": n,
            "thresholds": fit_thresholds(scores, levels),
        },
    )


def fit_thresholds(scores: Sequence[float], levels: Sequence[int]) -> list[float]:
    """For each boundary k (1..4) the cut that best separates level < k from level >= k."""
    cuts: list[float] = []
    lower = 0.0
    for k in range(1, 5):
        candidates = sorted({round(s, 3) for s in scores if s > lower} | {lower + 0.001})
        best, best_err = candidates[0], len(scores) + 1
        for cut in candidates:
            err = sum((s >= cut) != (lvl >= k) for s, lvl in zip(scores, levels, strict=True))
            if err < best_err:
                best, best_err = cut, err
        cuts.append(best)
        lower = best
    return cuts
