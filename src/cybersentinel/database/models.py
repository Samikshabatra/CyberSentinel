"""Database models.

Only what an analyst needs for incident history is persisted. Raw input is
stored truncated and redacted, and indicators are stored separately so history
queries ("has this IP appeared before?") do not require scanning raw text.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for all CyberSentinel tables."""


class Incident(Base):
    """One analysed incident."""

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    input_type: Mapped[str] = mapped_column(String(32), default="alert")
    #: Redacted and truncated. Never the full raw submission.
    input_preview: Mapped[str] = mapped_column(Text, default="")
    input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    attack_type: Mapped[str] = mapped_column(String(64), default="Unknown", index=True)
    severity: Mapped[str] = mapped_column(String(16), default="UNKNOWN", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    risk_score: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    likelihood: Mapped[int | None] = mapped_column(Integer, nullable=True)
    impact: Mapped[int | None] = mapped_column(Integer, nullable=True)

    approval_status: Mapped[str] = mapped_column(String(16), default="NOT_REQUIRED", index=True)
    approval_required: Mapped[bool] = mapped_column(default=False)
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    is_correlated: Mapped[bool] = mapped_column(default=False)
    model_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    #: Full structured report as produced by the report agent.
    report: Mapped[dict] = mapped_column(JSON, default=dict)
    mitre_techniques: Mapped[list] = mapped_column(JSON, default=list)
    recommendations: Mapped[list] = mapped_column(JSON, default=list)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    errors: Mapped[list] = mapped_column(JSON, default=list)

    indicators: Mapped[list[IncidentIndicator]] = relationship(
        back_populates="incident", cascade="all, delete-orphan", lazy="selectin"
    )
    approvals: Mapped[list[ApprovalAudit]] = relationship(
        back_populates="incident", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (Index("ix_incidents_attack_severity", "attack_type", "severity"),)


class IncidentIndicator(Base):
    """An observable extracted from an incident, used for history lookups."""

    __tablename__ = "incident_indicators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_pk: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16), index=True)
    value: Mapped[str] = mapped_column(String(255), index=True)

    incident: Mapped[Incident] = relationship(back_populates="indicators")

    __table_args__ = (Index("ix_indicator_kind_value", "kind", "value"),)


class ApprovalAudit(Base):
    """Audit trail of every human-in-the-loop decision."""

    __tablename__ = "approval_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_pk: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"))
    decision: Mapped[str] = mapped_column(String(16))
    decided_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    incident: Mapped[Incident] = relationship(back_populates="approvals")
