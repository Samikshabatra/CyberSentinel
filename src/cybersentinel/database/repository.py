"""Incident repository.

All database access goes through this class so persistence stays testable and
the rest of the system never writes SQL. Storage is deliberately minimal: a
redacted input preview rather than the raw submission, and indicators stored
separately so history questions are answerable without text scanning.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cybersentinel.database.models import ApprovalAudit, Incident, IncidentIndicator
from cybersentinel.utils.logging import get_logger, redact

logger = get_logger(__name__)

#: Indicator classes worth persisting for cross-incident lookups.
PERSISTED_INDICATORS: tuple[str, ...] = ("ips", "urls", "domains", "emails", "hashes", "users", "hosts")

#: Preview length for the stored input. Enough for an analyst to recognise the
#: event, short enough not to become a secondary data store of raw logs.
PREVIEW_CHARS = 500


@dataclass
class IncidentSummary:
    """Lightweight incident row for list views."""

    incident_id: str
    created_at: datetime
    attack_type: str
    severity: str
    confidence: float
    risk_score: int | None
    risk_level: str | None
    approval_status: str
    input_type: str
    input_preview: str
    is_correlated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "created_at": self.created_at.isoformat(),
            "attack_type": self.attack_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "approval_status": self.approval_status,
            "input_type": self.input_type,
            "input_preview": self.input_preview,
            "is_correlated": self.is_correlated,
        }


def _summarise(incident: Incident) -> IncidentSummary:
    return IncidentSummary(
        incident_id=incident.incident_id,
        created_at=incident.created_at,
        attack_type=incident.attack_type,
        severity=incident.severity,
        confidence=incident.confidence,
        risk_score=incident.risk_score,
        risk_level=incident.risk_level,
        approval_status=incident.approval_status,
        input_type=incident.input_type,
        input_preview=incident.input_preview,
        is_correlated=incident.is_correlated,
    )


class IncidentRepository:
    """CRUD and history queries for incidents."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # --- writes ------------------------------------------------------------
    def save_analysis(
        self,
        state: dict[str, Any],
        thread_id: str | None = None,
    ) -> Incident:
        """Persist (or update) an incident from a completed workflow state."""
        incident_id = str(state.get("incident_id") or "")
        if not incident_id:
            raise ValueError("state has no incident_id")

        report = state.get("final_report") or {}
        analysis = state.get("threat_analysis") or {}
        risk = state.get("risk_assessment") or {}
        approval = state.get("approval") or {}
        correlation = state.get("correlation") or {}
        metrics = state.get("metrics") or {}
        raw_input = str(state.get("input_text") or "")

        incident = self.get(incident_id) or Incident(incident_id=incident_id)

        incident.run_id = state.get("run_id")
        incident.thread_id = thread_id or state.get("incident_id")
        incident.input_type = str(state.get("input_type") or "alert")
        incident.input_preview = redact(raw_input, PREVIEW_CHARS)
        incident.input_hash = hashlib.sha256(raw_input.encode("utf-8")).hexdigest()

        incident.attack_type = str(analysis.get("attack_type") or "Unknown")
        incident.severity = str(report.get("severity") or analysis.get("severity") or "UNKNOWN")
        incident.confidence = float(analysis.get("confidence") or 0.0)

        incident.risk_score = risk.get("risk_score")
        incident.risk_level = risk.get("risk_level")
        incident.likelihood = risk.get("likelihood")
        incident.impact = risk.get("impact")

        incident.approval_status = str(approval.get("decision") or "NOT_REQUIRED")
        incident.approval_required = bool(approval.get("required", False))
        incident.approved_by = approval.get("decided_by")
        decided_at = approval.get("decided_at")
        incident.approved_at = _parse_datetime(decided_at)

        incident.is_correlated = bool(correlation.get("is_correlated", False))
        incident.model_source = metrics.get("model_source") or analysis.get("model_source")
        incident.latency_seconds = metrics.get("total_latency_seconds")

        incident.report = report
        incident.mitre_techniques = [
            technique.get("technique_id")
            for technique in (state.get("mitre_mapping") or {}).get("techniques", [])
        ]
        incident.recommendations = state.get("response_recommendations") or []
        incident.sources = report.get("sources") or []
        incident.errors = list(state.get("errors") or [])

        if incident.id is None:
            self.session.add(incident)
            self.session.flush()

        self._replace_indicators(incident, state.get("indicators") or {})
        self.session.flush()
        logger.info(f"incident persisted: {incident_id} ({incident.attack_type}/{incident.severity})")
        return incident

    def _replace_indicators(self, incident: Incident, indicators: dict[str, list[str]]) -> None:
        for existing in list(incident.indicators):
            self.session.delete(existing)
        incident.indicators.clear()

        for kind in PERSISTED_INDICATORS:
            for value in indicators.get(kind, [])[:25]:
                incident.indicators.append(
                    IncidentIndicator(kind=kind, value=str(value)[:255].lower())
                )

    def record_approval(
        self,
        incident_id: str,
        decision: str,
        decided_by: str | None = None,
        note: str | None = None,
        reason: str | None = None,
    ) -> Incident | None:
        """Append an approval audit entry and update the incident status."""
        incident = self.get(incident_id)
        if incident is None:
            return None

        incident.approval_status = decision
        incident.approved_by = decided_by
        incident.approved_at = datetime.now(UTC)
        incident.approvals.append(
            ApprovalAudit(
                decision=decision, decided_by=decided_by, note=note, reason=reason
            )
        )
        self.session.flush()
        return incident

    # --- reads -------------------------------------------------------------
    def get(self, incident_id: str) -> Incident | None:
        return self.session.execute(
            select(Incident).where(Incident.incident_id == incident_id)
        ).scalar_one_or_none()

    def list_incidents(
        self,
        limit: int = 50,
        offset: int = 0,
        severity: str | None = None,
        attack_type: str | None = None,
        approval_status: str | None = None,
    ) -> list[IncidentSummary]:
        statement = select(Incident).order_by(Incident.created_at.desc())
        if severity:
            statement = statement.where(Incident.severity == severity)
        if attack_type:
            statement = statement.where(Incident.attack_type == attack_type)
        if approval_status:
            statement = statement.where(Incident.approval_status == approval_status)

        rows = self.session.execute(statement.limit(limit).offset(offset)).scalars().all()
        return [_summarise(row) for row in rows]

    def pending_approvals(self, limit: int = 50) -> list[IncidentSummary]:
        """Incidents waiting on an analyst decision."""
        rows = (
            self.session.execute(
                select(Incident)
                .where(Incident.approval_status == "PENDING")
                .order_by(Incident.created_at.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return [_summarise(row) for row in rows]

    def find_by_indicator(
        self,
        value: str,
        kind: str | None = None,
        limit: int = 20,
        exclude_incident_id: str | None = None,
    ) -> list[IncidentSummary]:
        """Answer "has this indicator appeared in previous incidents?"."""
        statement = (
            select(Incident)
            .join(IncidentIndicator)
            .where(IncidentIndicator.value == value.strip().lower())
        )
        if kind:
            statement = statement.where(IncidentIndicator.kind == kind)
        if exclude_incident_id:
            statement = statement.where(Incident.incident_id != exclude_incident_id)

        rows = (
            self.session.execute(
                statement.order_by(Incident.created_at.desc()).limit(limit)
            )
            .scalars()
            .unique()
            .all()
        )
        return [_summarise(row) for row in rows]

    def find_similar(
        self,
        attack_type: str,
        limit: int = 10,
        exclude_incident_id: str | None = None,
    ) -> list[IncidentSummary]:
        """Answer "have we seen this attack pattern before?"."""
        statement = select(Incident).where(Incident.attack_type == attack_type)
        if exclude_incident_id:
            statement = statement.where(Incident.incident_id != exclude_incident_id)
        rows = (
            self.session.execute(statement.order_by(Incident.created_at.desc()).limit(limit))
            .scalars()
            .all()
        )
        return [_summarise(row) for row in rows]

    def history_for_indicators(
        self,
        indicators: dict[str, Sequence[str]],
        exclude_incident_id: str | None = None,
        limit_per_value: int = 5,
    ) -> list[dict[str, Any]]:
        """Look up every extracted indicator against incident history."""
        matches: list[dict[str, Any]] = []
        for kind in PERSISTED_INDICATORS:
            for value in list(indicators.get(kind, []))[:10]:
                previous = self.find_by_indicator(
                    value, kind, limit_per_value, exclude_incident_id
                )
                if previous:
                    matches.append(
                        {
                            "kind": kind,
                            "value": value,
                            "incident_count": len(previous),
                            "incidents": [summary.to_dict() for summary in previous],
                        }
                    )
        return matches

    # --- aggregates --------------------------------------------------------
    def metrics(self, days: int | None = None) -> dict[str, Any]:
        """Dashboard aggregates."""
        statement = select(Incident)
        if days:
            cutoff = datetime.now(UTC) - timedelta(days=days)
            statement = statement.where(Incident.created_at >= cutoff)

        incidents = self.session.execute(statement).scalars().all()
        total = len(incidents)

        by_severity: dict[str, int] = {}
        by_attack: dict[str, int] = {}
        by_approval: dict[str, int] = {}
        latencies: list[float] = []

        for incident in incidents:
            by_severity[incident.severity] = by_severity.get(incident.severity, 0) + 1
            by_attack[incident.attack_type] = by_attack.get(incident.attack_type, 0) + 1
            by_approval[incident.approval_status] = by_approval.get(incident.approval_status, 0) + 1
            if incident.latency_seconds is not None:
                latencies.append(incident.latency_seconds)

        return {
            "total_incidents": total,
            "critical_incidents": by_severity.get("CRITICAL", 0),
            "high_incidents": by_severity.get("HIGH", 0),
            "pending_approvals": by_approval.get("PENDING", 0),
            "correlated_incidents": sum(1 for incident in incidents if incident.is_correlated),
            "by_severity": by_severity,
            "by_attack_type": by_attack,
            "by_approval_status": by_approval,
            "average_latency_seconds": (
                round(sum(latencies) / len(latencies), 3) if latencies else None
            ),
        }

    def count(self) -> int:
        return int(self.session.execute(select(func.count(Incident.id))).scalar_one())


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
