"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from cybersentinel.schemas.analysis import IncidentReport, RetrievedDocument


class AnalyzeRequest(BaseModel):
    """Submit one security event (or a multi-event block) for analysis."""

    text: str = Field(
        min_length=1,
        max_length=20_000,
        description="Alert text, log excerpt, email, URL, or several events separated by blank "
        "lines or 'Event N:' headers.",
        examples=["47 failed SSH login attempts from 198.51.100.23 within 3 minutes."],
    )
    use_rag: bool = Field(default=True, description="Enable threat-intelligence retrieval.")
    use_llm_response: bool = Field(
        default=True, description="Use the model for recommendations (else the deterministic playbook)."
    )
    asset_criticality: int | None = Field(
        default=None, ge=1, le=5, description="Business criticality of the affected asset (1-5)."
    )
    incident_id: str | None = Field(
        default=None, max_length=64, description="Reuse an existing incident id."
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class BatchAnalyzeRequest(BaseModel):
    """Analyse several independent events, each as its own incident."""

    events: list[str] = Field(min_length=1, max_length=25)
    use_rag: bool = True


class PendingApproval(BaseModel):
    """What the analyst needs in order to decide, shown while a run is paused."""

    reason: str = Field(description="Why approval was triggered.")
    attack_type: str
    severity: str
    confidence: float
    risk_level: str | None = None
    risk_score: int | None = None
    evidence: list[str] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    high_impact_actions: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    mitre_techniques: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    """Result of one analysis."""

    incident_id: str
    run_id: str
    thread_id: str
    awaiting_approval: bool = Field(
        description="True when an analyst decision is required before the workflow completes."
    )
    report: IncidentReport | None = Field(
        default=None,
        description="Present once the workflow completes. Null while awaiting approval.",
    )
    pending_approval: PendingApproval | None = Field(
        default=None, description="Populated only while awaiting an analyst decision."
    )
    history_matches: list[dict[str, Any]] = Field(default_factory=list)
    node_path: list[str] = Field(default_factory=list, description="LangGraph nodes executed.")
    metrics: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]
    submitted: int
    analysed: int


class ApprovalRequest(BaseModel):
    """Analyst decision on a pending high-impact recommendation."""

    decision: Literal["APPROVED", "REJECTED", "ESCALATED"]
    decided_by: str | None = Field(default=None, max_length=128)
    note: str | None = Field(default=None, max_length=2000)


class IncidentSummaryResponse(BaseModel):
    incident_id: str
    created_at: str
    attack_type: str
    severity: str
    confidence: float
    risk_score: int | None = None
    risk_level: str | None = None
    approval_status: str
    input_type: str
    input_preview: str
    is_correlated: bool


class IncidentDetailResponse(BaseModel):
    incident_id: str
    created_at: str
    attack_type: str
    severity: str
    risk_score: int | None = None
    approval_status: str
    thread_id: str | None = None
    report: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ThreatIntelSearchResponse(BaseModel):
    query: str
    store: str
    latency_seconds: float
    documents: list[RetrievedDocument] = Field(default_factory=list)
    error: str | None = None


class ComponentHealth(BaseModel):
    name: str
    status: Literal["ok", "degraded", "unavailable"]
    detail: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    components: list[ComponentHealth] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    total_incidents: int
    critical_incidents: int
    high_incidents: int
    pending_approvals: int
    correlated_incidents: int
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_attack_type: dict[str, int] = Field(default_factory=dict)
    by_approval_status: dict[str, int] = Field(default_factory=dict)
    average_latency_seconds: float | None = None


class ErrorResponse(BaseModel):
    detail: str
    error_type: str | None = None
