"""Analysis and approval endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from cybersentinel.api.schemas.api_models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ApprovalRequest,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    PendingApproval,
)
from cybersentinel.service import AnalysisResult, AnalysisService, get_service
from cybersentinel.utils.logging import get_logger
from cybersentinel.utils.validation import InputValidationError

logger = get_logger(__name__)
router = APIRouter(tags=["analysis"])


def _pending_payload(result: AnalysisResult) -> PendingApproval | None:
    """Summarise a paused run so the analyst can decide without a second call."""
    if not result.awaiting_approval:
        return None

    state = result.state
    analysis = state.get("threat_analysis") or {}
    risk = state.get("risk_assessment") or {}
    approval = state.get("approval") or {}
    recommendations = state.get("response_recommendations") or []
    mapping = state.get("mitre_mapping") or {}

    return PendingApproval(
        reason=str(approval.get("reason") or "Analyst approval required."),
        attack_type=str(analysis.get("attack_type") or "Unknown"),
        severity=str(analysis.get("severity") or "UNKNOWN"),
        confidence=float(analysis.get("confidence") or 0.0),
        risk_level=risk.get("risk_level"),
        risk_score=risk.get("risk_score"),
        evidence=list(analysis.get("evidence") or []),
        recommendations=recommendations,
        high_impact_actions=[
            str(item.get("action"))
            for item in recommendations
            if item.get("high_impact")
        ],
        sources=list(state.get("retrieved_context") or [])[:5],
        mitre_techniques=[
            str(technique.get("technique_id")) for technique in mapping.get("techniques", [])
        ],
    )


def _to_response(result: AnalysisResult) -> AnalyzeResponse:
    return AnalyzeResponse(
        incident_id=result.incident_id,
        run_id=result.run_id,
        thread_id=result.thread_id,
        awaiting_approval=result.awaiting_approval,
        report=result.report or None,
        pending_approval=_pending_payload(result),
        history_matches=result.history_matches,
        node_path=result.node_path,
        metrics=result.state.get("metrics") or {},
        errors=result.errors,
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse a security event",
    description=(
        "Runs the full LangGraph workflow. When the assessed risk or a recommended action "
        "requires analyst sign-off, the run pauses and `awaiting_approval` is true; submit "
        "the decision to `/approval/{incident_id}` to resume it. No response action is ever "
        "executed by this service."
    ),
)
async def analyze(
    request: AnalyzeRequest,
    service: AnalysisService = Depends(get_service),
) -> AnalyzeResponse:
    try:
        result = service.analyze(
            request.text,
            use_rag=request.use_rag,
            use_llm_response=request.use_llm_response,
            asset_criticality=request.asset_criticality,
            incident_id=request.incident_id,
        )
    except InputValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("analysis failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"analysis failed: {type(exc).__name__}",
        ) from exc

    return _to_response(result)


@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    summary="Analyse several independent events",
    description=(
        "Each event is analysed as a separate incident. To correlate events into one "
        "incident, submit them together in a single `/analyze` request instead."
    ),
)
async def analyze_batch(
    request: BatchAnalyzeRequest,
    service: AnalysisService = Depends(get_service),
) -> BatchAnalyzeResponse:
    results = service.analyze_batch(request.events, use_rag=request.use_rag)
    return BatchAnalyzeResponse(
        results=[_to_response(result) for result in results],
        submitted=len(request.events),
        analysed=len(results),
    )


@router.post(
    "/approval/{incident_id}",
    response_model=AnalyzeResponse,
    summary="Submit a human-in-the-loop decision",
    description=(
        "Resumes a paused workflow. APPROVED completes the report as-is, REJECTED withdraws "
        "high-impact actions and re-analyses for investigative steps, ESCALATED records the "
        "incident for senior review. The service never executes the approved action itself."
    ),
)
async def submit_approval(
    incident_id: str,
    request: ApprovalRequest,
    service: AnalysisService = Depends(get_service),
) -> AnalyzeResponse:
    try:
        result = service.submit_decision(
            thread_id=incident_id,
            decision=request.decision,
            decided_by=request.decided_by,
            note=request.note,
        )
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"no analysis awaiting approval for incident '{incident_id}'",
        ) from exc
    except Exception as exc:
        logger.exception("approval failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"approval failed: {type(exc).__name__}",
        ) from exc

    return _to_response(result)
