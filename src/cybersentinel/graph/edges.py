"""Conditional routing functions.

These are the graph's real decision points - the reason LangGraph is used here
rather than a linear chain. Every router is a pure function of state, so the
routing decisions are deterministic and directly testable.
"""

from __future__ import annotations

from cybersentinel.cybersecurity.taxonomy import (
    ApprovalDecision,
    AttackType,
    InputType,
    Severity,
    normalise_severity,
    severity_at_least,
)
from cybersentinel.graph.state import CyberState
from cybersentinel.utils.config import get_settings

#: Maximum re-analysis passes after a rejection, so the cycle cannot spin.
MAX_REANALYSIS = 1


def route_after_classification(state: CyberState) -> str:
    """Route on input type.

    Multi-event submissions go through correlation-aware handling; every other
    type goes straight to detection. The distinction is what makes correlation a
    real branch rather than a step everything walks through.
    """
    input_type = state.get("input_type", InputType.ALERT.value)
    if input_type == InputType.MULTI_EVENT.value:
        return "multi_event"
    if input_type == InputType.EMAIL.value:
        return "email"
    if input_type == InputType.URL.value:
        return "url"
    if input_type == InputType.LOG.value:
        return "log"
    if input_type == InputType.VULNERABILITY.value:
        return "vulnerability"
    return "alert"


def route_after_detection(state: CyberState) -> str:
    """Skip retrieval when there is nothing to ground.

    A benign or unclassifiable event has no threat intelligence to look up.
    Querying anyway wastes latency and invites the model to attach techniques to
    a non-finding.
    """
    if not state.get("use_rag", True):
        return "skip_intel"

    analysis = state.get("threat_analysis") or {}
    attack_type = analysis.get("attack_type", AttackType.UNKNOWN.value)

    if attack_type in (AttackType.BENIGN.value, AttackType.UNKNOWN.value):
        return "skip_intel"
    return "intel"


def route_after_correlation(state: CyberState) -> str:
    """Always proceed to risk assessment; kept explicit for trace readability."""
    return "risk"


def route_after_risk(state: CyberState) -> str:
    """Decide whether the analyst must approve before recommendations are acted on.

    HIGH and CRITICAL risk always go through the approval gate. LOW and MEDIUM
    still reach the gate node itself, because a high-impact recommendation can
    require approval even at moderate risk - the gate makes that call.
    """
    risk = state.get("risk_assessment") or {}
    level = normalise_severity(risk.get("risk_level"))
    threshold = normalise_severity(get_settings().approval_severity_threshold)

    if level is Severity.UNKNOWN:
        return "low"
    return "high" if severity_at_least(level, threshold) else "low"


def route_after_gate(state: CyberState) -> str:
    """Stop for the analyst, or continue straight to the report."""
    approval = state.get("approval") or {}
    return "approval" if approval.get("required") else "report"


def route_after_approval(state: CyberState) -> str:
    """Branch on the analyst's decision.

    APPROVE  -> report as-is.
    REJECT   -> withdraw high-impact actions and re-analyse once.
    ESCALATE -> record and report without acting.
    PENDING  -> report as pending; the system never proceeds on its own.
    """
    decision = state.get("human_approval", ApprovalDecision.PENDING.value)

    if decision == ApprovalDecision.APPROVED.value:
        return "report"
    if decision == ApprovalDecision.REJECTED.value:
        if int(state.get("reanalysis_count", 0)) >= MAX_REANALYSIS:
            return "report"
        return "escalate"
    if decision == ApprovalDecision.ESCALATED.value:
        return "escalate"
    return "report"


def route_after_escalation(state: CyberState) -> str:
    """Re-analyse after a rejection; escalation goes straight to the report."""
    decision = state.get("human_approval", ApprovalDecision.PENDING.value)
    if decision == ApprovalDecision.REJECTED.value and int(state.get("reanalysis_count", 0)) <= MAX_REANALYSIS:
        return "reanalyse"
    return "report"
