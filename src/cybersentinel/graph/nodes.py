"""LangGraph node implementations.

Each node is a thin adapter: it reads what it needs from `CyberState`, calls an
agent, and returns a partial state update. Nodes never raise - a failure is
captured into ``errors`` and the workflow continues with a degraded but valid
state, which is what the blueprint's error-handling requirements demand.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC
from typing import Any

from cybersentinel.agents.correlation import correlate_events
from cybersentinel.agents.input_classifier import classify_input
from cybersentinel.agents.report import build_report
from cybersentinel.agents.response import recommend
from cybersentinel.agents.risk_assessment import assess
from cybersentinel.agents.threat_detector import aggregate_detections, detect_batch
from cybersentinel.agents.threat_intelligence import gather_intelligence
from cybersentinel.cybersecurity.risk import requires_human_approval
from cybersentinel.cybersecurity.taxonomy import (
    ApprovalDecision,
    AttackType,
    InputType,
    Severity,
    normalise_severity,
)
from cybersentinel.graph.state import CyberState
from cybersentinel.llm.model import LLMBackend, get_backend
from cybersentinel.rag.retriever import Retriever
from cybersentinel.schemas.analysis import (
    ApprovalRecord,
    CorrelationResult,
    MitreMapping,
    Recommendation,
    RiskAssessmentModel,
    SourceReference,
    ThreatAnalysis,
)
from cybersentinel.utils.config import get_settings
from cybersentinel.utils.logging import get_logger
from cybersentinel.utils.validation import extract_indicators, sanitize_text, split_events

logger = get_logger(__name__)


def _trace(node: str, started: float, status: str = "success", detail: str = "") -> dict[str, Any]:
    return {
        "node": node,
        "status": status,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "detail": detail,
    }


def _elapsed_since(started_at: str | None) -> float | None:
    """Wall-clock seconds since an ISO timestamp, or None if unparseable."""
    if not started_at:
        return None
    from datetime import datetime

    try:
        start = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    return round((datetime.now(UTC) - start).total_seconds(), 3)


def _log(state: CyberState, node: str, entry: dict[str, Any]) -> None:
    logger.info(
        "node completed",
        extra={
            "run_id": state.get("run_id"),
            "incident_id": state.get("incident_id"),
            "node": node,
            "status": entry["status"],
            "latency_s": entry["latency_seconds"],
        },
    )


def node_guard(node_name: str) -> Callable[..., Any]:
    """Decorator: turn any unhandled node exception into a recorded error."""

    def decorator(function: Callable[[CyberState], dict[str, Any]]) -> Callable[[CyberState], dict[str, Any]]:
        def wrapper(state: CyberState) -> dict[str, Any]:
            started = time.perf_counter()
            try:
                update = function(state)
                entry = update.pop("_trace", None) or _trace(node_name, started)
                _log(state, node_name, entry)
                return {**update, "node_trace": [entry]}
            except Exception as exc:
                message = f"{node_name}: {type(exc).__name__}: {exc}"
                logger.exception(
                    "node failed",
                    extra={
                        "run_id": state.get("run_id"),
                        "incident_id": state.get("incident_id"),
                        "node": node_name,
                        "status": "error",
                    },
                )
                return {
                    "errors": [message],
                    "node_trace": [_trace(node_name, started, "error", message)],
                }

        wrapper.__name__ = function.__name__
        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Shared resources
# --------------------------------------------------------------------------- #
class NodeContext:
    """Backend and retriever shared by all nodes in a workflow instance."""

    def __init__(self, backend: LLMBackend | None = None, retriever: Retriever | None = None) -> None:
        self._backend = backend
        self._retriever = retriever

    @property
    def backend(self) -> LLMBackend:
        if self._backend is None:
            self._backend = get_backend()
        return self._backend

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def make_input_classifier(context: NodeContext) -> Callable[[CyberState], dict[str, Any]]:
    @node_guard("input_classifier")
    def input_classifier(state: CyberState) -> dict[str, Any]:
        started = time.perf_counter()
        text = sanitize_text(state["input_text"])
        classification = classify_input(text)
        events = split_events(text) if classification.event_count > 1 else [text]

        return {
            "input_text": text,
            "input_type": classification.input_type.value,
            "events": events,
            "indicators": extract_indicators(text),
            "classification": classification.model_dump(mode="json"),
            "_trace": _trace(
                "input_classifier",
                started,
                detail=f"{classification.input_type.value} ({len(events)} event(s))",
            ),
        }

    return input_classifier


def make_threat_detector(context: NodeContext) -> Callable[[CyberState], dict[str, Any]]:
    @node_guard("threat_detector")
    def threat_detector(state: CyberState) -> dict[str, Any]:
        started = time.perf_counter()
        events = state.get("events") or [state["input_text"]]
        input_type = state.get("input_type", "alert")

        outcomes = detect_batch(events, input_type, backend=context.backend)
        analysis = aggregate_detections(outcomes) if len(outcomes) > 1 else outcomes[0].analysis

        per_event = [
            {
                "event_index": index,
                "analysis": outcome.analysis.model_dump(mode="json"),
                "valid_json": outcome.valid_json,
                "parse_strategy": outcome.parse_strategy,
                "latency_seconds": outcome.latency_seconds,
                "error": outcome.error,
            }
            for index, outcome in enumerate(outcomes)
        ]

        metrics = dict(state.get("metrics") or {})
        metrics["detection_latency_seconds"] = round(
            sum(outcome.latency_seconds for outcome in outcomes), 3
        )
        metrics["detection_prompt_tokens"] = sum(outcome.prompt_tokens for outcome in outcomes)
        metrics["detection_completion_tokens"] = sum(
            outcome.completion_tokens for outcome in outcomes
        )
        metrics["detection_json_valid"] = all(outcome.valid_json for outcome in outcomes)
        metrics["model_source"] = analysis.model_source

        errors = [outcome.error for outcome in outcomes if outcome.error]

        return {
            "threat_analysis": analysis.model_dump(mode="json"),
            "per_event_analyses": per_event,
            "metrics": metrics,
            **({"errors": errors} if errors else {}),
            "_trace": _trace(
                "threat_detector",
                started,
                detail=f"{analysis.attack_type.value} @ {analysis.confidence:.2f}",
            ),
        }

    return threat_detector


def make_threat_intelligence(context: NodeContext) -> Callable[[CyberState], dict[str, Any]]:
    @node_guard("threat_intelligence")
    def threat_intelligence(state: CyberState) -> dict[str, Any]:
        started = time.perf_counter()

        if not state.get("use_rag", True):
            return {
                "mitre_mapping": MitreMapping().model_dump(mode="json"),
                "_trace": _trace("threat_intelligence", started, detail="skipped (RAG disabled)"),
            }

        analysis = ThreatAnalysis.model_validate(state["threat_analysis"])
        outcome = gather_intelligence(
            analysis,
            state.get("input_text", ""),
            retriever=context.retriever,
            backend=context.backend,
        )

        metrics = dict(state.get("metrics") or {})
        metrics["rag_latency_seconds"] = outcome.latency_seconds
        metrics["retrieved_documents"] = len(outcome.retrieval.documents)
        metrics["rag_store"] = outcome.retrieval.store
        metrics["rejected_claims"] = len(outcome.mapping.rejected_claims)
        metrics["catalogue_fallback"] = outcome.used_catalogue_fallback

        return {
            "retrieved_context": [
                document.model_dump(mode="json") for document in outcome.retrieval.documents
            ],
            "context_text": outcome.context_text,
            "mitre_mapping": outcome.mapping.model_dump(mode="json"),
            "metrics": metrics,
            **({"errors": [f"threat_intelligence: {outcome.error}"]} if outcome.error else {}),
            "_trace": _trace(
                "threat_intelligence",
                started,
                detail=f"{len(outcome.retrieval.documents)} doc(s), "
                f"{len(outcome.mapping.techniques)} grounded technique(s)",
            ),
        }

    return threat_intelligence


def make_correlation(context: NodeContext) -> Callable[[CyberState], dict[str, Any]]:
    @node_guard("correlation")
    def correlation(state: CyberState) -> dict[str, Any]:
        started = time.perf_counter()
        events = state.get("events") or []

        if len(events) < 2:
            result = CorrelationResult(
                summary="Single event submitted; correlation is not applicable."
            )
            return {
                "correlation": result.model_dump(mode="json"),
                "_trace": _trace("correlation", started, detail="skipped (single event)"),
            }

        per_event = state.get("per_event_analyses") or []
        attack_types = [
            ThreatAnalysis.model_validate(entry["analysis"]).attack_type for entry in per_event
        ]
        evidence = [
            ThreatAnalysis.model_validate(entry["analysis"]).evidence for entry in per_event
        ]
        # Guard against a detector failure leaving fewer analyses than events.
        while len(attack_types) < len(events):
            attack_types.append(AttackType.UNKNOWN)
            evidence.append([])

        result = correlate_events(events, attack_types, evidence, backend=context.backend)

        return {
            "correlation": result.model_dump(mode="json"),
            "correlated_incidents": [
                stage.model_dump(mode="json") for stage in result.attack_chain
            ],
            "_trace": _trace(
                "correlation",
                started,
                detail=f"correlated={result.is_correlated} stages={len(result.attack_chain)}",
            ),
        }

    return correlation


def make_risk_assessment(context: NodeContext) -> Callable[[CyberState], dict[str, Any]]:
    @node_guard("risk_assessment")
    def risk_assessment(state: CyberState) -> dict[str, Any]:
        started = time.perf_counter()
        analysis = ThreatAnalysis.model_validate(state["threat_analysis"])
        correlation_data = state.get("correlation") or {}
        correlation = (
            CorrelationResult.model_validate(correlation_data)
            if correlation_data
            else CorrelationResult()
        )

        risk = assess(analysis, correlation, state.get("asset_criticality"))

        return {
            "risk_assessment": risk.model_dump(mode="json"),
            "_trace": _trace(
                "risk_assessment",
                started,
                detail=f"{risk.risk_level.value} (score {risk.risk_score})",
            ),
        }

    return risk_assessment


def make_response(context: NodeContext) -> Callable[[CyberState], dict[str, Any]]:
    @node_guard("response_recommendation")
    def response_recommendation(state: CyberState) -> dict[str, Any]:
        started = time.perf_counter()
        analysis = ThreatAnalysis.model_validate(state["threat_analysis"])
        risk = RiskAssessmentModel.model_validate(state["risk_assessment"])
        mapping_data = state.get("mitre_mapping") or {}
        mapping = MitreMapping.model_validate(mapping_data) if mapping_data else MitreMapping()

        recommendations = recommend(
            analysis,
            risk,
            mapping,
            backend=context.backend,
            use_llm=state.get("use_llm_response", True),
        )

        # A rejected recommendation set means the analyst wants a different
        # approach: drop the disruptive options and keep investigation only.
        if state.get("human_approval") == ApprovalDecision.REJECTED.value:
            recommendations = [item for item in recommendations if not item.high_impact]

        return {
            "response_recommendations": [item.model_dump(mode="json") for item in recommendations],
            "_trace": _trace(
                "response_recommendation",
                started,
                detail=f"{len(recommendations)} recommendation(s)",
            ),
        }

    return response_recommendation


def make_approval_gate(context: NodeContext) -> Callable[[CyberState], dict[str, Any]]:
    """Decide whether analyst approval is required, before any interrupt."""

    @node_guard("approval_gate")
    def approval_gate(state: CyberState) -> dict[str, Any]:
        started = time.perf_counter()
        settings = get_settings()

        # The gate is reached a second time after a rejection sends the incident
        # back for re-analysis. Asking for approval again would loop forever, so
        # a decision that has already been given is final for this run.
        decided = state.get("human_approval") in (
            ApprovalDecision.APPROVED.value,
            ApprovalDecision.REJECTED.value,
            ApprovalDecision.ESCALATED.value,
        )
        if decided or int(state.get("reanalysis_count", 0)) > 0:
            existing = dict(state.get("approval") or {})
            existing["required"] = False
            return {
                "approval": existing,
                "_trace": _trace(
                    "approval_gate",
                    started,
                    detail=f"decision already recorded ({state.get('human_approval')})",
                ),
            }

        risk = RiskAssessmentModel.model_validate(state["risk_assessment"])
        recommendations = [
            Recommendation.model_validate(item)
            for item in state.get("response_recommendations") or []
        ]

        required, reason = requires_human_approval(
            risk.risk_level,
            [item.action for item in recommendations if item.high_impact],
            threshold=normalise_severity(settings.approval_severity_threshold),
        )

        record = ApprovalRecord(
            decision=ApprovalDecision.PENDING if required else ApprovalDecision.NOT_REQUIRED,
            required=required,
            reason=reason,
        )

        return {
            "approval": record.model_dump(mode="json"),
            "human_approval": record.decision.value,
            "_trace": _trace(
                "approval_gate", started, detail=f"required={required}"
            ),
        }

    return approval_gate


def make_human_approval(context: NodeContext) -> Callable[[CyberState], dict[str, Any]]:
    """Apply the analyst's decision.

    The graph is compiled with ``interrupt_before=["human_approval"]``, so
    execution stops before this node runs and resumes only once a decision has
    been written into the state. Nothing is executed automatically either way -
    the decision only controls which recommendations are presented.
    """

    @node_guard("human_approval")
    def human_approval(state: CyberState) -> dict[str, Any]:
        started = time.perf_counter()
        from datetime import datetime

        decision_value = state.get("human_approval", ApprovalDecision.PENDING.value)
        try:
            decision = ApprovalDecision(decision_value)
        except ValueError:
            decision = ApprovalDecision.PENDING

        existing = state.get("approval") or {}
        record = ApprovalRecord(
            decision=decision,
            required=bool(existing.get("required", True)),
            reason=str(existing.get("reason", "")),
            decided_by=state.get("approved_by"),
            decided_at=datetime.now(UTC) if decision is not ApprovalDecision.PENDING else None,
            note=state.get("approval_note"),
        )

        return {
            "approval": record.model_dump(mode="json"),
            "human_approval": decision.value,
            "_trace": _trace("human_approval", started, detail=decision.value),
        }

    return human_approval


def make_escalation(context: NodeContext) -> Callable[[CyberState], dict[str, Any]]:
    """Handle a rejected or escalated analysis.

    Rejection is not a dead end: it is recorded, the disruptive recommendations
    are dropped, and the incident is routed back for a second pass. The loop is
    bounded by ``reanalysis_count``.
    """

    @node_guard("escalation")
    def escalation(state: CyberState) -> dict[str, Any]:
        started = time.perf_counter()
        count = int(state.get("reanalysis_count", 0)) + 1
        decision = state.get("human_approval", ApprovalDecision.PENDING.value)

        note = (
            "Analyst rejected the proposed response. High-impact actions have been withdrawn "
            "and the incident has been re-analysed for investigative next steps."
            if decision == ApprovalDecision.REJECTED.value
            else "Analyst escalated the incident for senior review. No action has been taken."
        )

        return {
            "reanalysis_count": count,
            "messages": [{"role": "system", "content": note}],
            "_trace": _trace("escalation", started, detail=f"{decision} (pass {count})"),
        }

    return escalation


def make_report(context: NodeContext) -> Callable[[CyberState], dict[str, Any]]:
    @node_guard("incident_report")
    def incident_report(state: CyberState) -> dict[str, Any]:
        started = time.perf_counter()

        analysis = ThreatAnalysis.model_validate(state["threat_analysis"])
        mapping_data = state.get("mitre_mapping") or {}
        mapping = MitreMapping.model_validate(mapping_data) if mapping_data else MitreMapping()
        correlation_data = state.get("correlation") or {}
        correlation = (
            CorrelationResult.model_validate(correlation_data)
            if correlation_data
            else CorrelationResult()
        )
        risk_data = state.get("risk_assessment") or {}
        risk = RiskAssessmentModel.model_validate(risk_data) if risk_data else None
        approval_data = state.get("approval") or {}
        approval = (
            ApprovalRecord.model_validate(approval_data) if approval_data else ApprovalRecord()
        )
        recommendations = [
            Recommendation.model_validate(item)
            for item in state.get("response_recommendations") or []
        ]
        sources = [
            SourceReference(
                source=document.get("source", "unknown"),
                document_id=document.get("document_id"),
                title=document.get("title"),
                url=document.get("url"),
                category=document.get("category"),
            )
            for document in state.get("retrieved_context") or []
        ]

        try:
            input_type = InputType(state.get("input_type", "alert"))
        except ValueError:
            input_type = InputType.ALERT

        total_latency = _elapsed_since(state.get("started_at"))

        report = build_report(
            incident_id=state["incident_id"],
            run_id=state.get("run_id", ""),
            input_type=input_type,
            analysis=analysis,
            mapping=mapping,
            correlation=correlation,
            risk=risk,
            recommendations=recommendations,
            approval=approval,
            sources=sources,
            errors=list(state.get("errors") or []),
            backend=context.backend,
            latency_seconds=total_latency,
        )

        metrics = dict(state.get("metrics") or {})
        metrics["total_latency_seconds"] = total_latency

        return {
            "final_report": report.model_dump(mode="json"),
            "metrics": metrics,
            "_trace": _trace(
                "incident_report",
                started,
                detail=f"{report.attack_type.value} / {report.severity.value}",
            ),
        }

    return incident_report


# Deduplicated severity import guard for callers of this module.
__all__ = [
    "NodeContext",
    "Severity",
    "make_approval_gate",
    "make_correlation",
    "make_escalation",
    "make_human_approval",
    "make_input_classifier",
    "make_report",
    "make_response",
    "make_risk_assessment",
    "make_threat_detector",
    "make_threat_intelligence",
]
