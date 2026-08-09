"""Structured output contracts.

Everything the system exposes externally - to the API, the UI, the database and
the evaluation harness - passes through these models. Validation constraints are
enforced here rather than trusted from the LLM: confidence is bounded, severity
comes from a fixed set, and identifier lists are normalised.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cybersentinel.cybersecurity.taxonomy import (
    ApprovalDecision,
    AttackType,
    InputType,
    Severity,
)

INSUFFICIENT_EVIDENCE = "Insufficient evidence"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Base model: unknown fields are rejected so malformed output fails loudly."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False, validate_assignment=True)


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
class SecurityEvent(StrictModel):
    """A single security event supplied for analysis."""

    event_id: str | None = Field(default=None, description="Caller-supplied identifier.")
    content: str = Field(min_length=1, description="Raw event text, log line, email or URL.")
    source: str | None = Field(default=None, description="Where the event came from, e.g. 'SIEM'.")
    timestamp: datetime | None = Field(default=None, description="When the event occurred.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("event content must not be empty")
        return cleaned


class InputClassification(StrictModel):
    """Result of the input-classifier node; drives conditional routing."""

    input_type: InputType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    indicators: list[str] = Field(default_factory=list)
    event_count: int = Field(default=1, ge=1)


# --------------------------------------------------------------------------- #
# Threat detection
# --------------------------------------------------------------------------- #
class ThreatAnalysis(StrictModel):
    """Output of the fine-tuned cybersecurity model for one event."""

    attack_type: AttackType = AttackType.UNKNOWN
    severity: Severity = Severity.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    candidate_techniques: list[str] = Field(
        default_factory=list,
        description="Technique ids proposed by the model; validated before use.",
    )
    reasoning: str = ""
    model_source: str = Field(
        default="unknown", description="Which backend produced this: mock | base | finetuned."
    )

    @field_validator("evidence", "candidate_techniques")
    @classmethod
    def _clean_list(cls, values: list[str]) -> list[str]:
        seen: list[str] = []
        for item in values:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return seen

    @model_validator(mode="after")
    def _unknown_requires_low_confidence(self) -> ThreatAnalysis:
        """An Unknown classification must not carry a confident severity claim."""
        if self.attack_type is AttackType.UNKNOWN and self.severity not in (
            Severity.UNKNOWN,
            Severity.LOW,
        ):
            object.__setattr__(self, "severity", Severity.UNKNOWN)
        return self

    @property
    def is_conclusive(self) -> bool:
        return self.attack_type is not AttackType.UNKNOWN and bool(self.evidence)


# --------------------------------------------------------------------------- #
# Threat intelligence / RAG
# --------------------------------------------------------------------------- #
class SourceReference(StrictModel):
    """A citation. Never fabricated - always derived from retrieved metadata."""

    source: str = Field(description="Authority, e.g. 'MITRE ATT&CK', 'CWE', 'NVD'.")
    document_id: str | None = None
    title: str | None = None
    url: str | None = None
    category: str | None = None


class RetrievedDocument(StrictModel):
    """A chunk returned by the vector store together with its provenance."""

    content: str
    score: float
    source: str
    document_id: str | None = None
    title: str | None = None
    url: str | None = None
    category: str | None = None

    def to_reference(self) -> SourceReference:
        return SourceReference(
            source=self.source,
            document_id=self.document_id,
            title=self.title,
            url=self.url,
            category=self.category,
        )


class MitreMapping(StrictModel):
    """Grounded threat-intelligence mapping.

    `rejected_claims` records identifiers the model proposed that could not be
    grounded. Keeping them visible is what makes the hallucination evaluation
    possible - they are reported, never silently dropped.
    """

    techniques: list[dict[str, str]] = Field(default_factory=list)
    tactics: list[str] = Field(default_factory=list)
    cwe: list[dict[str, str]] = Field(default_factory=list)
    cve: list[dict[str, str]] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    grounded: bool = Field(
        default=False, description="True when at least one mapping is supported by retrieval."
    )


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
class AttackChainStage(StrictModel):
    """One stage of a possible multi-stage intrusion."""

    stage: str = Field(description="ATT&CK tactic name, e.g. 'Credential Access'.")
    tactic_id: str | None = None
    event_indices: list[int] = Field(default_factory=list)
    description: str = ""
    supporting_evidence: list[str] = Field(default_factory=list)


class CorrelationResult(StrictModel):
    """Relationships found across multiple events."""

    is_correlated: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    shared_indicators: dict[str, list[str]] = Field(
        default_factory=dict, description="Indicator type -> values shared across events."
    )
    attack_chain: list[AttackChainStage] = Field(default_factory=list)
    summary: str = ""
    caveat: str = Field(
        default="Chain is a hypothesis derived from shared indicators and ordering; "
        "it requires analyst validation.",
    )


# --------------------------------------------------------------------------- #
# Risk and response
# --------------------------------------------------------------------------- #
class RiskAssessmentModel(StrictModel):
    """Serialised form of the deterministic risk calculation."""

    likelihood: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    risk_score: int = Field(ge=1, le=25)
    risk_level: Severity
    likelihood_label: str
    impact_label: str
    rationale: list[str] = Field(default_factory=list)
    formula: str = "risk_score = likelihood x impact"

    @model_validator(mode="after")
    def _score_must_match_inputs(self) -> RiskAssessmentModel:
        expected = self.likelihood * self.impact
        if self.risk_score != expected:
            raise ValueError(
                f"risk_score {self.risk_score} does not equal likelihood x impact ({expected})"
            )
        return self


class Recommendation(StrictModel):
    """A defensive, non-destructive recommended action."""

    action: str = Field(min_length=1)
    rationale: str = ""
    priority: Severity = Severity.MEDIUM
    high_impact: bool = Field(
        default=False, description="True when the action changes production state."
    )
    requires_approval: bool = False


class ApprovalRecord(StrictModel):
    """Human-in-the-loop checkpoint state."""

    decision: ApprovalDecision = ApprovalDecision.NOT_REQUIRED
    required: bool = False
    reason: str = ""
    decided_by: str | None = None
    decided_at: datetime | None = None
    note: str | None = None


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
class IncidentReport(StrictModel):
    """Final structured, explainable incident report."""

    incident_id: str
    run_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    summary: str = ""
    input_type: InputType = InputType.ALERT
    attack_type: AttackType = AttackType.UNKNOWN
    severity: Severity = Severity.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    evidence: list[str] = Field(default_factory=list)
    reasoning: str = ""

    mitre: MitreMapping = Field(default_factory=MitreMapping)
    correlation: CorrelationResult = Field(default_factory=CorrelationResult)
    risk: RiskAssessmentModel | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    approval: ApprovalRecord = Field(default_factory=ApprovalRecord)
    sources: list[SourceReference] = Field(default_factory=list)

    explainability: dict[str, str] = Field(
        default_factory=dict,
        description="Answers to: what/why/evidence/confidence/sources/risk/next steps.",
    )
    errors: list[str] = Field(default_factory=list)
    latency_seconds: float | None = None
    disclaimer: str = (
        "AI-assisted analysis. Findings are probabilistic and require analyst validation. "
        "No response action is executed automatically."
    )

    @property
    def is_insufficient(self) -> bool:
        return self.attack_type is AttackType.UNKNOWN and not self.evidence
