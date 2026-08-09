"""Application service layer.

Sits between the API/UI and the workflow. Responsibilities:

* run (or resume) the LangGraph workflow,
* enrich the result with incident history from the database,
* persist the incident.

The graph itself has no database dependency. Keeping persistence here means the
workflow stays unit-testable without a database, while incident memory is still
available to every caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cybersentinel.database.connection import init_database, session_scope
from cybersentinel.database.repository import IncidentRepository
from cybersentinel.graph.workflow import CyberSentinelWorkflow, WorkflowRun, get_workflow
from cybersentinel.utils.logging import get_logger
from cybersentinel.utils.validation import InputValidationError, sanitize_text

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Everything a caller needs after one analysis."""

    incident_id: str
    run_id: str
    thread_id: str
    awaiting_approval: bool
    report: dict[str, Any]
    state: dict[str, Any]
    history_matches: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def node_path(self) -> list[str]:
        return [entry.get("node", "") for entry in self.state.get("node_trace") or []]

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "awaiting_approval": self.awaiting_approval,
            "report": self.report,
            "history_matches": self.history_matches,
            "node_path": self.node_path,
            "metrics": self.state.get("metrics") or {},
            "errors": self.errors,
        }


class AnalysisService:
    """Coordinates workflow execution, incident memory and persistence."""

    def __init__(
        self,
        workflow: CyberSentinelWorkflow | None = None,
        session_factory: Any | None = None,
        auto_init_database: bool = True,
    ) -> None:
        self.workflow = workflow or get_workflow()
        self._session_factory = session_factory
        if auto_init_database:
            try:
                init_database()
            except Exception as exc:
                logger.warning(f"database unavailable at startup: {type(exc).__name__}: {exc}")

    def _session(self):
        return session_scope(self._session_factory)

    # --- analysis ----------------------------------------------------------
    def analyze(
        self,
        text: str,
        use_rag: bool = True,
        use_llm_response: bool = True,
        asset_criticality: int | None = None,
        incident_id: str | None = None,
        persist: bool = True,
    ) -> AnalysisResult:
        """Analyse one submission end to end."""
        cleaned = sanitize_text(text)
        run = self.workflow.analyze(
            cleaned,
            incident_id=incident_id,
            use_rag=use_rag,
            use_llm_response=use_llm_response,
            asset_criticality=asset_criticality,
        )
        return self._finalise(run, persist=persist)

    def analyze_batch(
        self,
        texts: list[str],
        use_rag: bool = True,
        persist: bool = True,
    ) -> list[AnalysisResult]:
        """Analyse several independent submissions.

        Each text is analysed as its own incident. To correlate events instead,
        submit them as a single multi-event input.
        """
        results: list[AnalysisResult] = []
        for text in texts:
            try:
                results.append(self.analyze(text, use_rag=use_rag, persist=persist))
            except InputValidationError as exc:
                logger.warning(f"batch item rejected: {exc}")
        return results

    def submit_decision(
        self,
        thread_id: str,
        decision: str,
        decided_by: str | None = None,
        note: str | None = None,
        persist: bool = True,
    ) -> AnalysisResult:
        """Resume a paused workflow with an analyst decision."""
        run = self.workflow.submit_decision(thread_id, decision, decided_by, note)
        result = self._finalise(run, persist=persist)

        if persist:
            try:
                with self._session() as session:
                    repository = IncidentRepository(session)
                    repository.record_approval(
                        result.incident_id,
                        decision=decision,
                        decided_by=decided_by,
                        note=note,
                        reason=str((run.state.get("approval") or {}).get("reason") or ""),
                    )
            except Exception as exc:
                logger.warning(f"could not audit approval: {type(exc).__name__}: {exc}")

        return result

    def _finalise(self, run: WorkflowRun, persist: bool) -> AnalysisResult:
        history: list[dict[str, Any]] = []
        state = run.state

        if persist:
            try:
                with self._session() as session:
                    repository = IncidentRepository(session)
                    history = repository.history_for_indicators(
                        state.get("indicators") or {},
                        exclude_incident_id=run.incident_id,
                    )
                    similar = repository.find_similar(
                        str((state.get("threat_analysis") or {}).get("attack_type") or "Unknown"),
                        limit=5,
                        exclude_incident_id=run.incident_id,
                    )
                    if similar:
                        history.append(
                            {
                                "kind": "attack_pattern",
                                "value": (state.get("threat_analysis") or {}).get("attack_type"),
                                "incident_count": len(similar),
                                "incidents": [summary.to_dict() for summary in similar],
                            }
                        )

                    state = {**state, "history_matches": history}
                    repository.save_analysis(state, thread_id=run.thread_id)
            except Exception as exc:
                logger.warning(f"persistence failed: {type(exc).__name__}: {exc}")
                state = {**state, "history_matches": history}

        # History is returned alongside the report, not merged into it: the
        # report is a validated IncidentReport and must stay schema-clean.
        report = dict(state.get("final_report") or {})

        return AnalysisResult(
            incident_id=run.incident_id,
            run_id=run.run_id,
            thread_id=run.thread_id,
            awaiting_approval=run.awaiting_approval,
            report=report,
            state=state,
            history_matches=history,
            errors=run.errors,
        )

    # --- queries -----------------------------------------------------------
    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            incident = IncidentRepository(session).get(incident_id)
            if incident is None:
                return None
            return {
                "incident_id": incident.incident_id,
                "created_at": incident.created_at.isoformat(),
                "attack_type": incident.attack_type,
                "severity": incident.severity,
                "risk_score": incident.risk_score,
                "approval_status": incident.approval_status,
                "thread_id": incident.thread_id,
                "report": incident.report,
                "errors": incident.errors,
            }

    def list_incidents(self, **filters: Any) -> list[dict[str, Any]]:
        with self._session() as session:
            return [
                summary.to_dict()
                for summary in IncidentRepository(session).list_incidents(**filters)
            ]

    def pending_approvals(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._session() as session:
            return [
                summary.to_dict()
                for summary in IncidentRepository(session).pending_approvals(limit)
            ]

    def metrics(self, days: int | None = None) -> dict[str, Any]:
        with self._session() as session:
            return IncidentRepository(session).metrics(days)

    def search_indicator(self, value: str, kind: str | None = None) -> list[dict[str, Any]]:
        with self._session() as session:
            return [
                summary.to_dict()
                for summary in IncidentRepository(session).find_by_indicator(value, kind)
            ]


_SERVICE: AnalysisService | None = None


def get_service() -> AnalysisService:
    """Return the process-wide service instance."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AnalysisService()
    return _SERVICE


def reset_service() -> None:
    """Drop the cached service (used by tests)."""
    global _SERVICE
    _SERVICE = None
