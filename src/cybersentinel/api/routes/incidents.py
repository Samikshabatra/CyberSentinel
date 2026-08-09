"""Incident history, threat-intelligence search, metrics and health endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cybersentinel import __version__
from cybersentinel.api.schemas.api_models import (
    ComponentHealth,
    HealthResponse,
    IncidentDetailResponse,
    IncidentSummaryResponse,
    MetricsResponse,
    ThreatIntelSearchResponse,
)
from cybersentinel.rag.retriever import Retriever
from cybersentinel.service import AnalysisService, get_service
from cybersentinel.utils.config import get_settings
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["incidents"])

_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever:
    """Process-wide retriever for the search endpoint."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


@router.get("/health", response_model=HealthResponse, tags=["system"], summary="Service health")
async def health() -> HealthResponse:
    """Report the status of each dependency. Degraded is not failure.

    The application is designed to run with fallbacks (mock LLM backend, local
    vector store, SQLite), so a missing service degrades a component rather
    than taking the API down.
    """
    settings = get_settings()
    components: list[ComponentHealth] = []
    degraded = False

    # LLM backend
    try:
        from cybersentinel.llm.model import get_backend

        info = get_backend().info
        components.append(ComponentHealth(name="llm", status="ok", detail=info))
        if info.get("backend") == "mock":
            degraded = True
    except Exception as exc:
        degraded = True
        components.append(
            ComponentHealth(name="llm", status="unavailable", detail={"error": str(exc)})
        )

    # Vector store + embeddings
    try:
        detail: dict[str, Any] = get_retriever().health()
        available = bool(detail.get("available")) and int(detail.get("points", 0)) > 0
        components.append(
            ComponentHealth(name="rag", status="ok" if available else "degraded", detail=detail)
        )
        degraded = degraded or not available
    except Exception as exc:
        degraded = True
        components.append(
            ComponentHealth(name="rag", status="unavailable", detail={"error": str(exc)})
        )

    # Database
    try:
        from cybersentinel.database.connection import session_scope
        from cybersentinel.database.repository import IncidentRepository

        with session_scope() as session:
            count = IncidentRepository(session).count()
        components.append(
            ComponentHealth(
                name="database",
                status="ok",
                detail={"incidents": count, "url": settings.database_url.split("@")[-1]},
            )
        )
    except Exception as exc:
        degraded = True
        components.append(
            ComponentHealth(name="database", status="unavailable", detail={"error": str(exc)})
        )

    return HealthResponse(
        status="degraded" if degraded else "ok", version=__version__, components=components
    )


@router.get(
    "/incidents",
    response_model=list[IncidentSummaryResponse],
    summary="List stored incidents",
)
async def list_incidents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    severity: str | None = Query(default=None),
    attack_type: str | None = Query(default=None),
    approval_status: str | None = Query(default=None),
    service: AnalysisService = Depends(get_service),
) -> list[IncidentSummaryResponse]:
    rows = service.list_incidents(
        limit=limit,
        offset=offset,
        severity=severity,
        attack_type=attack_type,
        approval_status=approval_status,
    )
    return [IncidentSummaryResponse(**row) for row in rows]


@router.get(
    "/incidents/pending-approval",
    response_model=list[IncidentSummaryResponse],
    summary="Incidents awaiting an analyst decision",
)
async def pending_approvals(
    limit: int = Query(default=50, ge=1, le=200),
    service: AnalysisService = Depends(get_service),
) -> list[IncidentSummaryResponse]:
    return [IncidentSummaryResponse(**row) for row in service.pending_approvals(limit)]


@router.get(
    "/incidents/{incident_id}",
    response_model=IncidentDetailResponse,
    summary="Fetch one incident report",
)
async def get_incident(
    incident_id: str,
    service: AnalysisService = Depends(get_service),
) -> IncidentDetailResponse:
    incident = service.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"incident '{incident_id}' not found")
    return IncidentDetailResponse(**incident)


@router.get(
    "/threat-intelligence/search",
    response_model=ThreatIntelSearchResponse,
    tags=["threat-intelligence"],
    summary="Search the cybersecurity knowledge base",
)
async def search_threat_intelligence(
    q: str = Query(min_length=2, max_length=500, description="Search query."),
    top_k: int = Query(default=5, ge=1, le=20),
    retriever: Retriever = Depends(get_retriever),
) -> ThreatIntelSearchResponse:
    result = retriever.retrieve(q, top_k=top_k)
    return ThreatIntelSearchResponse(
        query=result.query,
        store=result.store,
        latency_seconds=result.latency_seconds,
        documents=result.documents,
        error=result.error,
    )


@router.get(
    "/indicators/search",
    response_model=list[IncidentSummaryResponse],
    summary="Find previous incidents involving an indicator",
    description="Answers questions such as: has this IP appeared in previous incidents?",
)
async def search_indicator(
    value: str = Query(min_length=1, max_length=255),
    kind: str | None = Query(default=None, description="ips, urls, domains, emails, hashes, users, hosts"),
    service: AnalysisService = Depends(get_service),
) -> list[IncidentSummaryResponse]:
    return [IncidentSummaryResponse(**row) for row in service.search_indicator(value, kind)]


@router.get("/metrics", response_model=MetricsResponse, tags=["system"], summary="Dashboard metrics")
async def metrics(
    days: int | None = Query(default=None, ge=1, le=365),
    service: AnalysisService = Depends(get_service),
) -> MetricsResponse:
    return MetricsResponse(**service.metrics(days))
