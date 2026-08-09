"""Incident report agent (LangGraph node 8).

Assembles the final structured report and answers the eight explainability
questions from the blueprint. The narrative summary may come from the model,
but every field of substance is carried through from earlier nodes - the report
agent adds no new findings.
"""

from __future__ import annotations

from cybersentinel.cybersecurity.taxonomy import ApprovalDecision, AttackType, InputType
from cybersentinel.llm.inference import generate
from cybersentinel.llm.model import LLMBackend
from cybersentinel.llm.prompts import build_report_messages
from cybersentinel.llm.structured_output import parse_json_object
from cybersentinel.schemas.analysis import (
    ApprovalRecord,
    CorrelationResult,
    IncidentReport,
    MitreMapping,
    Recommendation,
    RiskAssessmentModel,
    SourceReference,
    ThreatAnalysis,
)
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)

INSUFFICIENT_SUMMARY = (
    "The submitted input did not contain sufficient evidence to assert a threat category. "
    "No attack classification is claimed. An analyst should review the raw event and, where "
    "possible, supply surrounding log context or related events."
)


def _deterministic_summary(
    analysis: ThreatAnalysis,
    risk: RiskAssessmentModel | None,
    mapping: MitreMapping,
    correlation: CorrelationResult,
) -> str:
    if analysis.attack_type is AttackType.UNKNOWN and not analysis.evidence:
        return INSUFFICIENT_SUMMARY

    if analysis.attack_type is AttackType.BENIGN:
        detail = f" Observations: {'; '.join(analysis.evidence[:3])}." if analysis.evidence else ""
        return (
            "The submitted activity is consistent with normal operation and no threat is "
            f"indicated (model confidence {analysis.confidence:.2f}).{detail} "
            "No response action is recommended beyond retaining the event for baseline tuning. "
            "This assessment reflects the submitted evidence only."
        )

    parts = [
        f"Activity consistent with a probable {analysis.attack_type.value} "
        f"(model confidence {analysis.confidence:.2f})."
    ]
    if analysis.evidence:
        parts.append(f"Supporting evidence: {'; '.join(analysis.evidence[:3])}.")
    if risk:
        parts.append(
            f"Risk assessed as {risk.risk_level.value} (score {risk.risk_score} = "
            f"likelihood {risk.likelihood} x impact {risk.impact})."
        )
    if mapping.techniques:
        ids = ", ".join(technique["technique_id"] for technique in mapping.techniques[:3])
        parts.append(f"Retrieved threat intelligence supports MITRE ATT&CK {ids}.")
    elif analysis.attack_type not in (AttackType.BENIGN, AttackType.UNKNOWN):
        parts.append("No threat-intelligence mapping was supported by the retrieved sources.")
    if correlation.is_correlated:
        stages = " -> ".join(stage.stage for stage in correlation.attack_chain)
        parts.append(f"Events correlate into a possible chain: {stages}.")
    parts.append("Findings require analyst validation before any action is taken.")
    return " ".join(parts)


def _explainability(
    analysis: ThreatAnalysis,
    risk: RiskAssessmentModel | None,
    mapping: MitreMapping,
    recommendations: list[Recommendation],
    approval: ApprovalRecord,
) -> dict[str, str]:
    """Answer the eight explainability questions from blueprint section 23."""
    sources = (
        ", ".join(
            f"{technique['source']} {technique['technique_id']}"
            for technique in mapping.techniques[:3]
        )
        or "No supporting threat-intelligence sources were retrieved."
    )
    next_step = (
        recommendations[0].action
        if recommendations
        else "Collect additional context for the event."
    )
    if approval.required and approval.decision is ApprovalDecision.PENDING:
        next_step = f"Analyst approval required before: {next_step}"

    return {
        "what_was_detected": analysis.attack_type.value,
        "why_detected": analysis.reasoning or "No reasoning was produced by the model.",
        "evidence": "; ".join(analysis.evidence) or "No evidence extracted.",
        "confidence": f"{analysis.confidence:.2f} (model-reported, not a probability of harm)",
        "threat_intelligence_sources": sources,
        "mitre_techniques": ", ".join(
            technique["technique_id"] for technique in mapping.techniques
        )
        or "None grounded in retrieved sources.",
        "risk": (
            f"{risk.risk_level.value} (score {risk.risk_score})" if risk else "Not assessed."
        ),
        "next_steps": next_step,
    }


def build_report(
    incident_id: str,
    run_id: str,
    input_type: InputType,
    analysis: ThreatAnalysis,
    mapping: MitreMapping,
    correlation: CorrelationResult,
    risk: RiskAssessmentModel | None,
    recommendations: list[Recommendation],
    approval: ApprovalRecord,
    sources: list[SourceReference],
    errors: list[str] | None = None,
    backend: LLMBackend | None = None,
    use_llm: bool = True,
    latency_seconds: float | None = None,
) -> IncidentReport:
    """Assemble the final incident report."""
    summary = _deterministic_summary(analysis, risk, mapping, correlation)

    # For Unknown and Benign the deterministic wording is deliberately careful
    # about what is *not* being claimed; the model is not asked to rephrase it.
    if use_llm and analysis.attack_type not in (AttackType.UNKNOWN, AttackType.BENIGN):
        payload = {
            "input_type": input_type.value,
            "attack_type": analysis.attack_type.value,
            "severity": analysis.severity.value,
            "confidence": f"{analysis.confidence:.2f}",
            "evidence": "; ".join(analysis.evidence) or "none",
            "intel": ", ".join(t["technique_id"] for t in mapping.techniques) or "none grounded",
            "correlation": correlation.summary or "not applicable",
            "risk": f"{risk.risk_level.value} ({risk.risk_score})" if risk else "not assessed",
            "recommendations": "; ".join(item.action for item in recommendations[:4]) or "none",
            "approval": approval.decision.value,
        }
        result = generate(build_report_messages(payload), backend=backend)
        if result.ok:
            parsed = parse_json_object(result.text)
            model_summary = (parsed.data or {}).get("summary") if parsed.ok else None
            if isinstance(model_summary, str) and len(model_summary.strip()) > 60:
                summary = model_summary.strip()
        else:
            logger.warning(f"report model call failed: {result.error}")

    return IncidentReport(
        incident_id=incident_id,
        run_id=run_id,
        summary=summary,
        input_type=input_type,
        attack_type=analysis.attack_type,
        severity=risk.risk_level if risk else analysis.severity,
        confidence=analysis.confidence,
        evidence=analysis.evidence,
        reasoning=analysis.reasoning,
        mitre=mapping,
        correlation=correlation,
        risk=risk,
        recommendations=recommendations,
        approval=approval,
        sources=sources,
        explainability=_explainability(analysis, risk, mapping, recommendations, approval),
        errors=errors or [],
        latency_seconds=latency_seconds,
    )
