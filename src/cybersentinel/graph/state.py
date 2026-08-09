"""Typed workflow state.

`CyberState` is the single object every node reads from and writes to. Nodes
return *partial* updates; LangGraph merges them. Two channels use reducers so
that concurrent or repeated writes append rather than overwrite:

* ``errors``       - accumulated, never lost
* ``node_trace``   - the execution path, used for agent evaluation

Everything else is last-write-wins, which is what we want for the analysis
results that later nodes refine.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict


def append_list(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Reducer: concatenate list channels instead of replacing them."""
    return (left or []) + (right or [])


class NodeTrace(TypedDict, total=False):
    """One entry in the execution trace."""

    node: str
    status: str
    latency_seconds: float
    detail: str


class CyberState(TypedDict, total=False):
    """State shared by every node in the CyberSentinel workflow."""

    # --- identifiers ---
    run_id: str
    incident_id: str
    started_at: str

    # --- input ---
    input_text: str
    input_type: str
    events: list[str]
    indicators: dict[str, list[str]]

    # --- analysis ---
    classification: dict[str, Any]
    threat_analysis: dict[str, Any]
    per_event_analyses: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]
    context_text: str
    mitre_mapping: dict[str, Any]
    correlated_incidents: list[dict[str, Any]]
    correlation: dict[str, Any]
    history_matches: list[dict[str, Any]]
    risk_assessment: dict[str, Any]

    # --- response and approval ---
    response_recommendations: list[dict[str, Any]]
    human_approval: str
    approval: dict[str, Any]
    approval_note: str
    approved_by: str

    # --- output ---
    final_report: dict[str, Any]

    # --- bookkeeping ---
    messages: Annotated[list[dict[str, Any]], append_list]
    errors: Annotated[list[str], append_list]
    node_trace: Annotated[list[NodeTrace], append_list]
    metrics: dict[str, Any]

    # --- configuration carried through the run ---
    use_rag: bool
    use_llm_response: bool
    asset_criticality: int | None
    reanalysis_count: int


def new_incident_id() -> str:
    """Human-readable incident identifier: INC-YYYYMMDD-XXXXXX."""
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return f"INC-{stamp}-{uuid.uuid4().hex[:6].upper()}"


def initial_state(
    input_text: str,
    run_id: str,
    incident_id: str | None = None,
    use_rag: bool = True,
    use_llm_response: bool = True,
    asset_criticality: int | None = None,
) -> CyberState:
    """Build the starting state for one analysis run."""
    return CyberState(
        run_id=run_id,
        incident_id=incident_id or new_incident_id(),
        started_at=datetime.now(UTC).isoformat(),
        input_text=input_text,
        input_type="alert",
        events=[],
        indicators={},
        classification={},
        threat_analysis={},
        per_event_analyses=[],
        retrieved_context=[],
        context_text="",
        mitre_mapping={},
        correlated_incidents=[],
        correlation={},
        history_matches=[],
        risk_assessment={},
        response_recommendations=[],
        human_approval="PENDING",
        approval={},
        final_report={},
        messages=[],
        errors=[],
        node_trace=[],
        metrics={},
        use_rag=use_rag,
        use_llm_response=use_llm_response,
        asset_criticality=asset_criticality,
        reanalysis_count=0,
    )


def record_metric(state: CyberState, key: str, value: Any) -> dict[str, Any]:
    """Return an updated metrics dict (state is never mutated in place)."""
    metrics = dict(state.get("metrics") or {})
    metrics[key] = value
    return metrics
