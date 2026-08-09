"""Correlation agent (LangGraph node 4).

Wraps the deterministic correlation engine and, when a chain is found, asks the
model for a readable summary. The model may only phrase what the engine already
established - if its summary is unusable, the deterministic summary is used.
"""

from __future__ import annotations

from cybersentinel.cybersecurity.correlation import CorrelationOutcome, correlate
from cybersentinel.cybersecurity.taxonomy import AttackType
from cybersentinel.llm.inference import generate
from cybersentinel.llm.model import LLMBackend
from cybersentinel.llm.prompts import build_correlation_messages
from cybersentinel.llm.structured_output import parse_json_object
from cybersentinel.schemas.analysis import AttackChainStage, CorrelationResult
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)


def _deterministic_summary(outcome: CorrelationOutcome, events: list[str]) -> str:
    if not outcome.is_correlated:
        return (
            f"The {len(events)} submitted events do not share enough indicators to be "
            "treated as a single incident."
        )
    stages = " -> ".join(stage.stage for stage in outcome.chain)
    shared = outcome.shared_map
    shared_text = "; ".join(f"{kind}: {', '.join(values)}" for kind, values in shared.items())
    return (
        f"The events are consistent with a possible multi-stage intrusion progressing "
        f"{stages}. They are linked by {shared_text}. This is a hypothesis based on shared "
        "indicators and event ordering, and requires analyst validation."
    )


def correlate_events(
    events: list[str],
    attack_types: list[AttackType],
    evidence_per_event: list[list[str]] | None = None,
    backend: LLMBackend | None = None,
    use_llm: bool = True,
) -> CorrelationResult:
    """Correlate events and return the structured result."""
    outcome = correlate(events, attack_types, evidence_per_event)

    chain = [
        AttackChainStage(
            stage=stage.stage,
            tactic_id=stage.tactic_id,
            event_indices=stage.event_indices,
            description=stage.description,
            supporting_evidence=stage.supporting_evidence[:5],
        )
        for stage in outcome.chain
    ]

    summary = _deterministic_summary(outcome, events)

    if use_llm and outcome.is_correlated:
        indicators_text = "; ".join(
            f"{kind}: {', '.join(values)}" for kind, values in outcome.shared_map.items()
        ) or "none"
        detections_text = "; ".join(
            f"[{index}] {attack_type.value}" for index, attack_type in enumerate(attack_types)
        )
        result = generate(
            build_correlation_messages(events, indicators_text, detections_text), backend=backend
        )
        if result.ok:
            parsed = parse_json_object(result.text)
            model_summary = (parsed.data or {}).get("summary") if parsed.ok else None
            if isinstance(model_summary, str) and len(model_summary.strip()) > 40:
                summary = model_summary.strip()
        else:
            logger.warning(f"correlation model call failed: {result.error}")

    return CorrelationResult(
        is_correlated=outcome.is_correlated,
        confidence=outcome.confidence,
        shared_indicators=outcome.shared_map,
        attack_chain=chain,
        summary=summary,
    )
