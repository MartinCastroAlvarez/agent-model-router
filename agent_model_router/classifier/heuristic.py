"""The default classifier: keyword/structure rules for the task, a linear scorer for difficulty.

Deterministic, offline and explainable: every decision carries the features that moved it.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Protocol

from agent_model_router.classifier.features import extract
from agent_model_router.classifier.features import keyword_hits
from agent_model_router.classifier.linear import LinearScorer
from agent_model_router.config import BUCKETS
from agent_model_router.security import redact
from agent_model_router.view import RequestView

__all__ = [
    "TASKS",
    "Classifier",
    "Decision",
    "HeuristicClassifier",
    "bucket_for",
    "bucket_index",
    "classify_task",
    "guard_floor",
]

TASKS = ("chat", "edit", "plan", "debug", "review", "tool_followup")
_THRESHOLDS = (0.2, 0.4, 0.6, 0.8)


def bucket_for(difficulty: float, thresholds: tuple[float, ...] = _THRESHOLDS) -> str:
    for i, cut in enumerate(thresholds):
        if difficulty < cut:
            return BUCKETS[i]
    return BUCKETS[-1]


def bucket_index(bucket: str) -> int:
    return BUCKETS.index(bucket)


@dataclass(frozen=True)
class Decision:
    """What the classifier concluded about one request."""

    task: str
    difficulty: float
    bucket: str
    source: str
    reasons: tuple[str, ...] = ()
    features: dict[str, float] = field(default_factory=dict)
    contributions: tuple[tuple[str, float], ...] = ()
    secret: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "difficulty": round(self.difficulty, 3),
            "bucket": self.bucket,
            "source": self.source,
            "reasons": list(self.reasons),
            "features": {k: round(v, 3) for k, v in self.features.items() if v},
            "contributions": [[k, round(v, 3)] for k, v in self.contributions],
            "secret_detected": self.secret,
        }


class Classifier(Protocol):
    async def classify(self, view: RequestView) -> Decision: ...


def classify_task(view: RequestView, features: dict[str, float]) -> str:
    """Rule-based task type."""
    if view.tool_followup:
        return "tool_followup"
    text = view.last_user_text
    scores = {
        "debug": 2.0 * features["traceback"] + 1.5 * keyword_hits(text, "debug"),
        "review": 1.5 * keyword_hits(text, "review"),
        "plan": 1.2 * keyword_hits(text, "plan")
        + (0.8 if keyword_hits(text, "hard") and "design" in text.lower() else 0.0),
        "edit": 0.9 * keyword_hits(text, "edit") + (0.5 if features["files"] else 0.0),
        "chat": 0.9 if features["short_question"] else 0.4,
    }
    order = ("debug", "review", "plan", "edit", "chat")
    return max(order, key=lambda t: (scores[t], -order.index(t)))


def guard_floor(text: str) -> tuple[int, str | None]:
    """Sensitive topics can only raise the bucket (production, credentials, billing...)."""
    hits = keyword_hits(text, "sensitive")
    if hits >= 2:
        return bucket_index("hard"), f"sensitive topics ({hits}) -> at least hard"
    if hits == 1:
        return bucket_index("medium"), "sensitive topic -> at least medium"
    return 0, None


class HeuristicClassifier:
    """Rules for the task, :class:`LinearScorer` for the difficulty, guards that only raise."""

    def __init__(self, scorer: LinearScorer | None = None) -> None:
        self.scorer = scorer or LinearScorer.load()

    def decide(self, view: RequestView) -> Decision:
        features = extract(view)
        task = classify_task(view, features)
        difficulty = self.scorer.score(features)
        contributions = tuple(self.scorer.contributions(features)[:4])
        reasons = [f"task={task}", f"score={difficulty:.2f}"]
        reasons += [f"{k}{'+' if v > 0 else ''}{v:.2f}" for k, v in contributions[:3]]
        cuts = self.scorer.thresholds
        idx = bucket_index(bucket_for(difficulty, cuts))
        floor, why = guard_floor(view.last_user_text)
        if floor > idx:
            idx = floor
            difficulty = max(difficulty, cuts[floor - 1])
            reasons.append(str(why))
        _, secret = redact(view.last_user_text)
        if secret:
            reasons.append("secret-shaped text in the prompt")
        return Decision(
            task=task,
            difficulty=difficulty,
            bucket=BUCKETS[idx],
            source="heuristic",
            reasons=tuple(reasons),
            features=features,
            contributions=contributions,
            secret=secret,
        )

    async def classify(self, view: RequestView) -> Decision:
        return self.decide(view)
