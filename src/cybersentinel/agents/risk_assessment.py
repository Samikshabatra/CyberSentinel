"""Risk assessment agent (LangGraph node 5).

Pure deterministic computation - see `cybersecurity.risk`. Keeping the LLM out
of scoring is what makes the risk number reproducible and auditable.
"""

from __future__ import annotations

from cybersentinel.cybersecurity.risk import assess_risk
from cybersentinel.schemas.analysis import (
    CorrelationResult,
    RiskAssessmentModel,
    ThreatAnalysis,
)


def assess(
    analysis: ThreatAnalysis,
    correlation: CorrelationResult | None = None,
    asset_criticality: int | None = None,
) -> RiskAssessmentModel:
    """Compute the risk assessment for an incident."""
    corroborating = 0
    if correlation and correlation.is_correlated:
        # Each distinct kill-chain stage beyond the first corroborates the finding.
        corroborating = max(0, len(correlation.attack_chain))

    assessment = assess_risk(
        attack_type=analysis.attack_type,
        confidence=analysis.confidence,
        evidence_count=len(analysis.evidence),
        detected_severity=analysis.severity,
        corroborating_events=corroborating,
        asset_criticality=asset_criticality,
    )

    rationale = list(assessment.rationale)
    if correlation and correlation.is_correlated:
        stages = " -> ".join(stage.stage for stage in correlation.attack_chain)
        rationale.append(f"Correlated multi-stage activity observed: {stages}.")

    return RiskAssessmentModel(
        likelihood=assessment.likelihood,
        impact=assessment.impact,
        risk_score=assessment.risk_score,
        risk_level=assessment.risk_level,
        likelihood_label=assessment.likelihood_label,
        impact_label=assessment.impact_label,
        rationale=rationale,
    )
