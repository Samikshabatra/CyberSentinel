"""LangGraph workflow assembly and execution.

Graph shape:

    START -> input_classifier
                |  (conditional on input type)
                +-- multi_event --> threat_detector -> correlation -> ...
                +-- others ------> threat_detector -> ...
          threat_detector
                |  (conditional: skip retrieval for non-findings)
                +-- intel ------> threat_intelligence --> correlation
                +-- skip_intel -> correlation
          correlation -> risk_assessment -> response_recommendation -> approval_gate
                |  (conditional on approval requirement)
                +-- approval --> [INTERRUPT] human_approval
                |                     |
                |                     +-- APPROVE  --> incident_report
                |                     +-- REJECT   --> escalation --> response_recommendation
                |                     +-- ESCALATE --> escalation --> incident_report
                +-- report ----> incident_report -> END

The graph is compiled with a checkpointer and ``interrupt_before`` on the
human-approval node, so a run genuinely pauses and can be resumed later by the
API once an analyst has decided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from cybersentinel.graph import edges, nodes
from cybersentinel.graph.state import CyberState, initial_state, new_incident_id
from cybersentinel.llm.model import LLMBackend
from cybersentinel.rag.retriever import Retriever
from cybersentinel.utils.logging import get_logger, new_run_id
from cybersentinel.utils.validation import sanitize_text

logger = get_logger(__name__)

APPROVAL_NODE = "human_approval"


def build_graph(
    backend: LLMBackend | None = None,
    retriever: Retriever | None = None,
    checkpointer: Any | None = None,
    enable_interrupt: bool = True,
) -> Any:
    """Build and compile the CyberSentinel workflow."""
    context = nodes.NodeContext(backend=backend, retriever=retriever)
    graph: StateGraph = StateGraph(CyberState)

    graph.add_node("input_classifier", nodes.make_input_classifier(context))
    graph.add_node("threat_detector", nodes.make_threat_detector(context))
    graph.add_node("threat_intelligence", nodes.make_threat_intelligence(context))
    graph.add_node("correlation", nodes.make_correlation(context))
    graph.add_node("risk_assessment", nodes.make_risk_assessment(context))
    graph.add_node("response_recommendation", nodes.make_response(context))
    graph.add_node("approval_gate", nodes.make_approval_gate(context))
    graph.add_node(APPROVAL_NODE, nodes.make_human_approval(context))
    graph.add_node("escalation", nodes.make_escalation(context))
    graph.add_node("incident_report", nodes.make_report(context))

    graph.add_edge(START, "input_classifier")

    # Input type routing. Every branch reaches detection, but the branch taken is
    # recorded in the trace and is what the agent-routing evaluation measures.
    graph.add_conditional_edges(
        "input_classifier",
        edges.route_after_classification,
        {
            "multi_event": "threat_detector",
            "email": "threat_detector",
            "url": "threat_detector",
            "log": "threat_detector",
            "vulnerability": "threat_detector",
            "alert": "threat_detector",
        },
    )

    graph.add_conditional_edges(
        "threat_detector",
        edges.route_after_detection,
        {"intel": "threat_intelligence", "skip_intel": "correlation"},
    )

    graph.add_edge("threat_intelligence", "correlation")
    graph.add_conditional_edges(
        "correlation", edges.route_after_correlation, {"risk": "risk_assessment"}
    )
    graph.add_conditional_edges(
        "risk_assessment",
        edges.route_after_risk,
        {"high": "response_recommendation", "low": "response_recommendation"},
    )
    graph.add_edge("response_recommendation", "approval_gate")

    graph.add_conditional_edges(
        "approval_gate",
        edges.route_after_gate,
        {"approval": APPROVAL_NODE, "report": "incident_report"},
    )
    graph.add_conditional_edges(
        APPROVAL_NODE,
        edges.route_after_approval,
        {"report": "incident_report", "escalate": "escalation"},
    )
    graph.add_conditional_edges(
        "escalation",
        edges.route_after_escalation,
        {"reanalyse": "response_recommendation", "report": "incident_report"},
    )

    graph.add_edge("incident_report", END)

    saver = checkpointer if checkpointer is not None else InMemorySaver()
    return graph.compile(
        checkpointer=saver,
        interrupt_before=[APPROVAL_NODE] if enable_interrupt else None,
    )


@dataclass
class WorkflowRun:
    """Outcome of one workflow execution."""

    state: dict[str, Any]
    thread_id: str
    interrupted: bool
    incident_id: str
    run_id: str

    @property
    def report(self) -> dict[str, Any]:
        return self.state.get("final_report") or {}

    @property
    def awaiting_approval(self) -> bool:
        return self.interrupted

    @property
    def errors(self) -> list[str]:
        return list(self.state.get("errors") or [])


class CyberSentinelWorkflow:
    """Convenience wrapper around the compiled graph.

    Holds one checkpointer for the process so a paused run can be resumed by a
    later request using the same thread id.
    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        retriever: Retriever | None = None,
        checkpointer: Any | None = None,
        enable_interrupt: bool = True,
    ) -> None:
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.enable_interrupt = enable_interrupt
        self.graph = build_graph(
            backend=backend,
            retriever=retriever,
            checkpointer=self.checkpointer,
            enable_interrupt=enable_interrupt,
        )

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def analyze(
        self,
        input_text: str,
        incident_id: str | None = None,
        use_rag: bool = True,
        use_llm_response: bool = True,
        asset_criticality: int | None = None,
        thread_id: str | None = None,
    ) -> WorkflowRun:
        """Run the workflow until completion or until approval is required.

        Input is validated here rather than inside a node: an empty or
        unusable submission is a caller error, and failing at the boundary
        gives a clear message instead of an incident report about nothing.
        """
        input_text = sanitize_text(input_text)
        run_id = new_run_id()
        resolved_incident = incident_id or new_incident_id()
        resolved_thread = thread_id or resolved_incident

        state = initial_state(
            input_text=input_text,
            run_id=run_id,
            incident_id=resolved_incident,
            use_rag=use_rag,
            use_llm_response=use_llm_response,
            asset_criticality=asset_criticality,
        )

        config = self._config(resolved_thread)
        logger.info(
            "workflow started",
            extra={"run_id": run_id, "incident_id": resolved_incident, "node": "workflow"},
        )
        final = self.graph.invoke(state, config)
        interrupted = self._is_interrupted(config)

        return WorkflowRun(
            state=dict(final),
            thread_id=resolved_thread,
            interrupted=interrupted,
            incident_id=resolved_incident,
            run_id=run_id,
        )

    def submit_decision(
        self,
        thread_id: str,
        decision: str,
        decided_by: str | None = None,
        note: str | None = None,
    ) -> WorkflowRun:
        """Resume a paused run with the analyst's decision."""
        config = self._config(thread_id)
        snapshot = self.graph.get_state(config)
        if not snapshot.next:
            raise ValueError(f"no paused run for thread '{thread_id}'")

        self.graph.update_state(
            config,
            {
                "human_approval": decision,
                "approved_by": decided_by,
                "approval_note": note,
            },
        )
        final = self.graph.invoke(None, config)

        return WorkflowRun(
            state=dict(final),
            thread_id=thread_id,
            interrupted=self._is_interrupted(config),
            incident_id=str(final.get("incident_id", "")),
            run_id=str(final.get("run_id", "")),
        )

    def get_state(self, thread_id: str) -> dict[str, Any]:
        """Return the current persisted state for a thread."""
        snapshot = self.graph.get_state(self._config(thread_id))
        return dict(snapshot.values) if snapshot.values else {}

    def _is_interrupted(self, config: dict[str, Any]) -> bool:
        snapshot = self.graph.get_state(config)
        return bool(snapshot.next) and APPROVAL_NODE in snapshot.next

    def mermaid(self) -> str:
        """Return the graph as a Mermaid diagram (used in the docs and the UI)."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as exc:  # pragma: no cover - drawing is best-effort
            logger.warning(f"could not render graph diagram: {exc}")
            return ""


_WORKFLOW: CyberSentinelWorkflow | None = None


def get_workflow() -> CyberSentinelWorkflow:
    """Return the process-wide workflow instance."""
    global _WORKFLOW
    if _WORKFLOW is None:
        _WORKFLOW = CyberSentinelWorkflow()
    return _WORKFLOW


def reset_workflow() -> None:
    """Drop the cached workflow (used by tests)."""
    global _WORKFLOW
    _WORKFLOW = None
