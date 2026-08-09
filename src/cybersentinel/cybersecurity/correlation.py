"""Deterministic correlation engine.

Correlation is computed from observable facts - shared indicators and the
kill-chain position of each event's classification - rather than asked of the
model. The model is used afterwards only to phrase the hypothesis.

Confidence is built from explicit components so it can be defended:

* shared indicators across events (the strongest signal)
* ordering consistent with the ATT&CK kill chain
* number of distinct stages observed

A chain is always presented as a hypothesis. The engine never claims a
confirmed intrusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cybersentinel.cybersecurity.mitre import (
    CATEGORY_TECHNIQUE_HINTS,
    TACTIC_ORDER,
    TACTICS,
    get_technique,
)
from cybersentinel.cybersecurity.taxonomy import AttackType
from cybersentinel.utils.validation import extract_indicators

#: Indicator classes that meaningfully tie two events to one actor or target.
LINKING_INDICATORS: tuple[str, ...] = ("ips", "users", "hosts", "domains", "hashes", "emails")

#: Where each category sits in the kill chain, used to order a candidate chain.
CATEGORY_TACTIC: dict[AttackType, str] = {
    AttackType.RECONNAISSANCE: "TA0043",
    AttackType.PHISHING: "TA0001",
    AttackType.SQL_INJECTION: "TA0001",
    AttackType.XSS: "TA0001",
    AttackType.VULNERABILITY: "TA0001",
    AttackType.BRUTE_FORCE: "TA0006",
    AttackType.CREDENTIAL_ATTACK: "TA0006",
    AttackType.SUSPICIOUS_AUTH: "TA0001",
    AttackType.MALWARE: "TA0002",
    AttackType.PRIVILEGE_ESCALATION: "TA0004",
    AttackType.INSIDER_THREAT: "TA0009",
    AttackType.DATA_EXFILTRATION: "TA0010",
    AttackType.DDOS: "TA0040",
}


@dataclass
class SharedIndicator:
    """One indicator value observed in more than one event."""

    kind: str
    value: str
    event_indices: list[int] = field(default_factory=list)


@dataclass
class ChainStage:
    """One stage of the hypothesised chain."""

    stage: str
    tactic_id: str
    event_indices: list[int]
    description: str
    supporting_evidence: list[str] = field(default_factory=list)


@dataclass
class CorrelationOutcome:
    """Result of deterministic correlation."""

    is_correlated: bool
    confidence: float
    shared: list[SharedIndicator]
    chain: list[ChainStage]
    rationale: list[str] = field(default_factory=list)

    @property
    def shared_map(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for indicator in self.shared:
            grouped.setdefault(indicator.kind, []).append(indicator.value)
        return grouped


def find_shared_indicators(events: list[str]) -> list[SharedIndicator]:
    """Find indicator values appearing in two or more events."""
    per_event = [extract_indicators(event) for event in events]
    shared: list[SharedIndicator] = []

    for kind in LINKING_INDICATORS:
        occurrences: dict[str, list[int]] = {}
        for index, indicators in enumerate(per_event):
            for value in indicators.get(kind, []):
                occurrences.setdefault(value.lower(), []).append(index)
        for value, indices in occurrences.items():
            unique = sorted(set(indices))
            if len(unique) >= 2:
                shared.append(SharedIndicator(kind=kind, value=value, event_indices=unique))

    return shared


def _chain_ordering_score(stage_positions: list[int]) -> tuple[float, str]:
    """Score how well the observed order follows the kill chain."""
    if len(stage_positions) < 2:
        return 0.0, "Fewer than two distinct stages; ordering carries no information."

    ascending = sum(
        1 for first, second in zip(stage_positions, stage_positions[1:], strict=False) if second >= first
    )
    total = len(stage_positions) - 1
    ratio = ascending / total

    if ratio == 1.0:
        return 0.25, "Event order matches ATT&CK kill-chain progression exactly."
    if ratio >= 0.5:
        return 0.12, f"Event order is partially consistent with the kill chain ({ascending}/{total})."
    return 0.0, "Event order does not follow kill-chain progression."


def build_chain(
    events: list[str],
    attack_types: list[AttackType],
    evidence_per_event: list[list[str]] | None = None,
) -> list[ChainStage]:
    """Group events into kill-chain stages, preserving submission order."""
    evidence_per_event = evidence_per_event or [[] for _ in events]
    stages: dict[str, ChainStage] = {}

    for index, attack_type in enumerate(attack_types):
        tactic_id = CATEGORY_TACTIC.get(attack_type)
        if tactic_id is None:
            continue

        if tactic_id not in stages:
            stages[tactic_id] = ChainStage(
                stage=TACTICS[tactic_id],
                tactic_id=tactic_id,
                event_indices=[],
                description="",
                supporting_evidence=[],
            )

        stage = stages[tactic_id]
        stage.event_indices.append(index)
        summary = " ".join(events[index].split())[:160]
        stage.description = (
            f"{attack_type.value} activity observed."
            if not stage.description
            else stage.description
        )
        stage.supporting_evidence.extend(
            item for item in (evidence_per_event[index] or [summary]) if item not in stage.supporting_evidence
        )

    ordered = sorted(
        stages.values(),
        key=lambda stage: TACTIC_ORDER.index(stage.tactic_id)
        if stage.tactic_id in TACTIC_ORDER
        else len(TACTIC_ORDER),
    )
    return ordered


def correlate(
    events: list[str],
    attack_types: list[AttackType],
    evidence_per_event: list[list[str]] | None = None,
) -> CorrelationOutcome:
    """Correlate multiple events into a hypothesised attack chain."""
    if len(events) < 2:
        return CorrelationOutcome(
            is_correlated=False,
            confidence=0.0,
            shared=[],
            chain=[],
            rationale=["Correlation requires at least two events."],
        )

    shared = find_shared_indicators(events)
    chain = build_chain(events, attack_types, evidence_per_event)
    rationale: list[str] = []
    confidence = 0.0

    if shared:
        linked_events = {index for indicator in shared for index in indicator.event_indices}
        coverage = len(linked_events) / len(events)
        contribution = round(0.45 * coverage, 3)
        confidence += contribution
        kinds = sorted({indicator.kind for indicator in shared})
        rationale.append(
            f"{len(shared)} shared indicator(s) ({', '.join(kinds)}) link "
            f"{len(linked_events)} of {len(events)} events (+{contribution})."
        )
    else:
        rationale.append("No indicator is shared between events (+0.0).")

    positions = [
        TACTIC_ORDER.index(stage.tactic_id)
        for stage in chain
        if stage.tactic_id in TACTIC_ORDER
    ]
    ordering_score, ordering_reason = _chain_ordering_score(positions)
    confidence += ordering_score
    rationale.append(f"{ordering_reason} (+{ordering_score})")

    distinct_stages = len(chain)
    if distinct_stages >= 3:
        confidence += 0.2
        rationale.append(f"{distinct_stages} distinct kill-chain stages observed (+0.2).")
    elif distinct_stages == 2:
        confidence += 0.1
        rationale.append("Two distinct kill-chain stages observed (+0.1).")

    confidence = round(min(confidence, 0.9), 3)

    # A chain hypothesis requires a real link between events, not just ordering.
    is_correlated = bool(shared) and distinct_stages >= 2
    if not is_correlated:
        rationale.append(
            "Not treated as a single incident: a shared indicator and at least two "
            "distinct stages are both required."
        )

    return CorrelationOutcome(
        is_correlated=is_correlated,
        confidence=confidence,
        shared=shared,
        chain=chain if is_correlated else [],
        rationale=rationale,
    )


def chain_techniques(chain: list[ChainStage], attack_types: list[AttackType]) -> list[str]:
    """Candidate technique ids implied by the stages in a chain.

    These are candidates only; grounding still happens in the intelligence
    agent against retrieved context.
    """
    techniques: list[str] = []
    for stage in chain:
        for index in stage.event_indices:
            if index >= len(attack_types):
                continue
            for technique_id in CATEGORY_TECHNIQUE_HINTS.get(attack_types[index], ()):
                if get_technique(technique_id) and technique_id not in techniques:
                    techniques.append(technique_id)
    return techniques
