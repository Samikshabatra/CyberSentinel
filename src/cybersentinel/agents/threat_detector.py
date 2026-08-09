"""Threat detection agent (LangGraph node 2).

This is the only place the fine-tuned cybersecurity model is asked to make a
classification judgement. Threat intelligence is explicitly *not* its job -
that belongs to retrieval (blueprint section 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cybersentinel.cybersecurity.taxonomy import AttackType, Severity
from cybersentinel.llm.inference import analyze_event
from cybersentinel.llm.model import LLMBackend
from cybersentinel.schemas.analysis import ThreatAnalysis
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionOutcome:
    """A detection plus the diagnostics needed for evaluation."""

    analysis: ThreatAnalysis
    latency_seconds: float = 0.0
    valid_json: bool = False
    parse_strategy: str = "unknown"
    missing_fields: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def detect_threat(
    event_text: str,
    input_type: str = "alert",
    backend: LLMBackend | None = None,
) -> DetectionOutcome:
    """Classify one event with the cybersecurity model."""
    analysis, generation, parse = analyze_event(event_text, input_type, backend=backend)

    return DetectionOutcome(
        analysis=analysis,
        latency_seconds=generation.latency_seconds,
        valid_json=parse.valid_json,
        parse_strategy=parse.strategy,
        missing_fields=parse.missing_fields,
        prompt_tokens=generation.approx_prompt_tokens,
        completion_tokens=generation.approx_completion_tokens,
        error=generation.error or parse.error,
    )


def detect_batch(
    events: list[str],
    input_type: str = "alert",
    backend: LLMBackend | None = None,
) -> list[DetectionOutcome]:
    """Classify several events, one detection per event."""
    return [detect_threat(event, input_type, backend) for event in events]


def aggregate_detections(outcomes: list[DetectionOutcome]) -> ThreatAnalysis:
    """Reduce per-event detections into one incident-level analysis.

    The most severe conclusive detection wins, because an incident is
    characterised by its worst confirmed component. Evidence from every event is
    retained so the report explains the whole picture, and confidence is the
    winning detection's own confidence - it is never inflated by agreement.
    """
    conclusive = [outcome for outcome in outcomes if outcome.analysis.is_conclusive]
    if not conclusive:
        merged_evidence: list[str] = []
        for outcome in outcomes:
            merged_evidence.extend(outcome.analysis.evidence)
        return ThreatAnalysis(
            attack_type=AttackType.UNKNOWN,
            severity=Severity.UNKNOWN,
            confidence=max((o.analysis.confidence for o in outcomes), default=0.0),
            evidence=merged_evidence,
            reasoning="No individual event produced a conclusive classification.",
            model_source=outcomes[0].analysis.model_source if outcomes else "unknown",
        )

    from cybersentinel.cybersecurity.taxonomy import SEVERITY_ORDER

    primary = max(
        conclusive,
        key=lambda outcome: (
            SEVERITY_ORDER[outcome.analysis.severity],
            outcome.analysis.confidence,
        ),
    )

    evidence: list[str] = []
    techniques: list[str] = []
    for outcome in conclusive:
        for item in outcome.analysis.evidence:
            if item not in evidence:
                evidence.append(item)
        for technique in outcome.analysis.candidate_techniques:
            if technique not in techniques:
                techniques.append(technique)

    other_labels = sorted(
        {
            outcome.analysis.attack_type.value
            for outcome in conclusive
            if outcome.analysis.attack_type is not primary.analysis.attack_type
        }
    )
    reasoning = primary.analysis.reasoning
    if other_labels:
        reasoning += (
            f" Other events in this submission were classified as: {', '.join(other_labels)}."
        )

    return ThreatAnalysis(
        attack_type=primary.analysis.attack_type,
        severity=primary.analysis.severity,
        confidence=primary.analysis.confidence,
        evidence=evidence,
        candidate_techniques=techniques,
        reasoning=reasoning,
        model_source=primary.analysis.model_source,
    )
