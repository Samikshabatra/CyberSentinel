"""Metric computation.

Classification metrics come from scikit-learn where it is available, with a
dependency-free fallback so evaluation still runs in a minimal environment.

Two choices worth stating for the write-up:

* **Macro F1 is the headline metric.** The test set is balanced by construction,
  but macro F1 keeps a rare category from being hidden by a common one, which is
  the failure mode that matters in a SOC.
* **Structural metrics are reported separately from classification metrics.** A
  model that classifies well but emits unparseable JSON is useless in a
  pipeline, so JSON validity and field completeness are first-class results
  rather than footnotes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cybersentinel.cybersecurity.taxonomy import AttackType, Severity


@dataclass
class ClassificationMetrics:
    """Standard classification results plus per-class detail."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    support: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "support": self.support,
        }


def _manual_classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]
) -> ClassificationMetrics:
    """Compute metrics without scikit-learn."""
    index = {label: position for position, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0] * size for _ in range(size)]

    for true, predicted in zip(y_true, y_pred, strict=True):
        if true in index and predicted in index:
            matrix[index[true]][index[predicted]] += 1

    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: dict[str, int] = {}

    for label in labels:
        position = index[label]
        true_positive = matrix[position][position]
        false_positive = sum(matrix[row][position] for row in range(size)) - true_positive
        false_negative = sum(matrix[position]) - true_positive
        support = sum(matrix[position])

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        supports[label] = support
        if support > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    total = len(y_true)
    correct = sum(1 for true, predicted in zip(y_true, y_pred, strict=True) if true == predicted)
    weighted_f1 = (
        sum(per_class[label]["f1"] * supports[label] for label in labels) / total if total else 0.0
    )

    return ClassificationMetrics(
        accuracy=round(correct / total, 4) if total else 0.0,
        macro_precision=round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        macro_recall=round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        macro_f1=round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        weighted_f1=round(weighted_f1, 4),
        per_class=per_class,
        confusion_matrix=matrix,
        labels=labels,
        support=supports,
    )


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str] | None = None,
    macro_over_observed: bool = True,
) -> ClassificationMetrics:
    """Compute classification metrics for a set of predictions.

    ``labels`` fixes the confusion-matrix axes (normally the full taxonomy, so
    the matrix shape is comparable across runs). Macro averages are computed
    over the labels actually observed unless ``macro_over_observed`` is False:
    averaging in classes that never appear in a subset would drag macro F1 down
    for a reason that has nothing to do with model quality.
    """
    if not y_true:
        return ClassificationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    resolved_labels = labels or sorted(set(y_true) | set(y_pred))
    observed = [label for label in resolved_labels if label in set(y_true) | set(y_pred)]
    macro_labels = observed if macro_over_observed and observed else resolved_labels

    try:
        from sklearn.metrics import (
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
    except ImportError:
        return _manual_classification_metrics(y_true, y_pred, macro_labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=resolved_labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=macro_labels, average="macro", zero_division=0
    )

    per_class = {
        label: {
            "precision": round(float(precision[position]), 4),
            "recall": round(float(recall[position]), 4),
            "f1": round(float(f1[position]), 4),
            "support": int(support[position]),
        }
        for position, label in enumerate(resolved_labels)
    }

    correct = sum(1 for true, predicted in zip(y_true, y_pred, strict=True) if true == predicted)

    return ClassificationMetrics(
        accuracy=round(correct / len(y_true), 4),
        macro_precision=round(float(macro_precision), 4),
        macro_recall=round(float(macro_recall), 4),
        macro_f1=round(float(macro_f1), 4),
        weighted_f1=round(
            float(f1_score(y_true, y_pred, labels=macro_labels, average="weighted", zero_division=0)),
            4,
        ),
        per_class=per_class,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=resolved_labels).tolist(),
        labels=resolved_labels,
        support={label: int(support[position]) for position, label in enumerate(resolved_labels)},
    )


@dataclass
class StructuralMetrics:
    """How well the model respects the output contract."""

    total: int
    json_valid_rate: float
    json_repaired_rate: float
    json_failed_rate: float
    field_completeness: float
    evidence_present_rate: float
    severity_accuracy: float
    mean_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "json_valid_rate": self.json_valid_rate,
            "json_repaired_rate": self.json_repaired_rate,
            "json_failed_rate": self.json_failed_rate,
            "field_completeness": self.field_completeness,
            "evidence_present_rate": self.evidence_present_rate,
            "severity_accuracy": self.severity_accuracy,
            "mean_confidence": self.mean_confidence,
        }


REQUIRED_FIELDS = ("attack_type", "severity", "confidence", "evidence")


def structural_metrics(predictions: list[dict[str, Any]]) -> StructuralMetrics:
    """Compute output-contract metrics over prediction records.

    Each record is expected to carry: ``parse_strategy``, ``missing_fields``,
    ``evidence``, ``severity``, ``expected_severity`` and ``confidence``.
    """
    total = len(predictions)
    if total == 0:
        return StructuralMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    strategies = Counter(record.get("parse_strategy", "failed") for record in predictions)
    strict_valid = strategies["direct"] + strategies["extracted"]

    completeness = sum(
        (len(REQUIRED_FIELDS) - len(record.get("missing_fields", []))) / len(REQUIRED_FIELDS)
        for record in predictions
    )
    with_evidence = sum(1 for record in predictions if record.get("evidence"))

    severity_pairs = [
        (record.get("severity"), record.get("expected_severity"))
        for record in predictions
        if record.get("expected_severity")
    ]
    severity_correct = sum(1 for predicted, expected in severity_pairs if predicted == expected)

    return StructuralMetrics(
        total=total,
        json_valid_rate=round(strict_valid / total, 4),
        json_repaired_rate=round(strategies["repaired"] / total, 4),
        json_failed_rate=round((strategies["failed"] + strategies["empty"]) / total, 4),
        field_completeness=round(completeness / total, 4),
        evidence_present_rate=round(with_evidence / total, 4),
        severity_accuracy=round(severity_correct / len(severity_pairs), 4) if severity_pairs else 0.0,
        mean_confidence=round(
            sum(float(record.get("confidence", 0.0)) for record in predictions) / total, 4
        ),
    )


def severity_within_one(predicted: str, expected: str) -> bool:
    """True when severities differ by at most one band.

    Reported alongside exact severity accuracy because a HIGH/CRITICAL mix-up is
    operationally very different from a LOW/CRITICAL one, and exact-match alone
    hides that distinction.
    """
    from cybersentinel.cybersecurity.taxonomy import SEVERITY_ORDER, normalise_severity

    left = SEVERITY_ORDER[normalise_severity(predicted)]
    right = SEVERITY_ORDER[normalise_severity(expected)]
    return abs(left - right) <= 1


def latency_summary(latencies: Sequence[float]) -> dict[str, float]:
    """Mean, median and p95 latency."""
    if not latencies:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}

    ordered = sorted(latencies)
    count = len(ordered)
    median = (
        ordered[count // 2]
        if count % 2
        else (ordered[count // 2 - 1] + ordered[count // 2]) / 2
    )
    p95_index = max(0, min(count - 1, int(round(0.95 * count)) - 1))

    return {
        "mean": round(sum(ordered) / count, 4),
        "median": round(median, 4),
        "p95": round(ordered[p95_index], 4),
        "max": round(ordered[-1], 4),
    }


def default_labels() -> list[str]:
    """The full closed label set, so absent classes still appear in the report."""
    return [label.value for label in AttackType]


def default_severities() -> list[str]:
    return [level.value for level in Severity]
