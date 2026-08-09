"""LangGraph workflow tests: routing, state, human-in-the-loop, error handling."""

from __future__ import annotations

import pytest

from cybersentinel.cybersecurity.taxonomy import ApprovalDecision, AttackType, Severity
from cybersentinel.graph import edges
from cybersentinel.graph.state import CyberState, initial_state, new_incident_id
from tests.conftest import (
    BENIGN_EVENT,
    BRUTE_FORCE_EVENT,
    MULTI_EVENT,
    PHISHING_EMAIL,
    VAGUE_EVENT,
)


def node_path(state: dict) -> list[str]:
    return [entry["node"] for entry in state.get("node_trace", [])]


# --------------------------------------------------------------------------- #
# Routing functions (pure)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("input_type", "expected"),
    [
        ("multi_event", "multi_event"),
        ("email", "email"),
        ("url", "url"),
        ("log", "log"),
        ("vulnerability", "vulnerability"),
        ("alert", "alert"),
    ],
)
def test_classification_routing(input_type, expected):
    assert edges.route_after_classification(CyberState(input_type=input_type)) == expected


def test_detection_routing_skips_retrieval_for_non_findings():
    for label in (AttackType.BENIGN.value, AttackType.UNKNOWN.value):
        state = CyberState(threat_analysis={"attack_type": label, "evidence": ["x"]}, use_rag=True)
        assert edges.route_after_detection(state) == "skip_intel"


def test_detection_routing_retrieves_for_findings():
    state = CyberState(
        threat_analysis={"attack_type": AttackType.BRUTE_FORCE.value, "evidence": ["x"]},
        use_rag=True,
    )
    assert edges.route_after_detection(state) == "intel"


def test_detection_routing_respects_disabled_rag():
    state = CyberState(
        threat_analysis={"attack_type": AttackType.BRUTE_FORCE.value, "evidence": ["x"]},
        use_rag=False,
    )
    assert edges.route_after_detection(state) == "skip_intel"


@pytest.mark.parametrize(
    ("risk_level", "expected"),
    [("CRITICAL", "high"), ("HIGH", "high"), ("MEDIUM", "low"), ("LOW", "low")],
)
def test_risk_routing(risk_level, expected):
    assert edges.route_after_risk(CyberState(risk_assessment={"risk_level": risk_level})) == expected


def test_gate_routing():
    assert edges.route_after_gate(CyberState(approval={"required": True})) == "approval"
    assert edges.route_after_gate(CyberState(approval={"required": False})) == "report"


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (ApprovalDecision.APPROVED.value, "report"),
        (ApprovalDecision.REJECTED.value, "escalate"),
        (ApprovalDecision.ESCALATED.value, "escalate"),
        (ApprovalDecision.PENDING.value, "report"),
    ],
)
def test_approval_routing(decision, expected):
    state = CyberState(human_approval=decision, reanalysis_count=0)
    assert edges.route_after_approval(state) == expected


def test_rejection_loop_is_bounded():
    """A rejected analysis re-runs once, then must terminate."""
    exhausted = CyberState(
        human_approval=ApprovalDecision.REJECTED.value, reanalysis_count=edges.MAX_REANALYSIS
    )
    assert edges.route_after_approval(exhausted) == "report"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def test_incident_id_format():
    incident_id = new_incident_id()
    assert incident_id.startswith("INC-")
    assert len(incident_id.split("-")) == 3


def test_initial_state_defaults():
    state = initial_state("some event", run_id="abc")
    assert state["input_text"] == "some event"
    assert state["human_approval"] == "PENDING"
    assert state["errors"] == []
    assert state["reanalysis_count"] == 0


# --------------------------------------------------------------------------- #
# End-to-end workflow
# --------------------------------------------------------------------------- #
def test_high_risk_run_pauses_for_approval(workflow):
    run = workflow.analyze(BRUTE_FORCE_EVENT)

    assert run.awaiting_approval
    assert run.state["risk_assessment"]["risk_level"] in ("HIGH", "CRITICAL")
    assert run.state["approval"]["required"]
    # The report must not exist yet: the analyst has not decided.
    assert not run.state.get("final_report")
    assert "incident_report" not in node_path(run.state)


def test_approval_resumes_and_produces_report(workflow):
    run = workflow.analyze(BRUTE_FORCE_EVENT)
    assert run.awaiting_approval

    resumed = workflow.submit_decision(run.thread_id, "APPROVED", decided_by="analyst")

    assert not resumed.awaiting_approval
    assert resumed.report["approval"]["decision"] == "APPROVED"
    assert resumed.report["approval"]["decided_by"] == "analyst"
    assert "incident_report" in node_path(resumed.state)


def test_rejection_withdraws_disruptive_actions(workflow):
    run = workflow.analyze(BRUTE_FORCE_EVENT)
    resumed = workflow.submit_decision(run.thread_id, "REJECTED", decided_by="analyst")

    assert resumed.report["approval"]["decision"] == "REJECTED"
    assert not any(item["high_impact"] for item in resumed.report["recommendations"])
    assert resumed.report["recommendations"], "investigative steps must remain"
    assert "escalation" in node_path(resumed.state)


def test_escalation_is_recorded_without_action(workflow):
    run = workflow.analyze(BRUTE_FORCE_EVENT)
    resumed = workflow.submit_decision(run.thread_id, "ESCALATED", decided_by="analyst")

    assert resumed.report["approval"]["decision"] == "ESCALATED"
    assert "escalation" in node_path(resumed.state)


def test_decision_on_unknown_thread_raises(workflow):
    with pytest.raises(ValueError):
        workflow.submit_decision("no-such-thread", "APPROVED")


def test_benign_event_completes_without_approval(straight_through_workflow):
    run = straight_through_workflow.analyze(BENIGN_EVENT)

    assert not run.awaiting_approval
    assert run.report["attack_type"] == AttackType.BENIGN.value
    assert run.report["risk"]["risk_level"] == Severity.LOW.value
    assert run.state["approval"]["decision"] == ApprovalDecision.NOT_REQUIRED.value
    assert "threat_intelligence" not in node_path(run.state)


def test_vague_event_returns_unknown_without_claims(straight_through_workflow):
    run = straight_through_workflow.analyze(VAGUE_EVENT)

    assert run.report["attack_type"] == AttackType.UNKNOWN.value
    assert run.report["mitre"]["techniques"] == []
    assert run.report["evidence"] == []


def test_multi_event_correlation_and_chain(workflow):
    run = workflow.analyze(MULTI_EVENT)
    state = run.state

    assert state["input_type"] == "multi_event"
    assert len(state["events"]) == 4
    assert state["correlation"]["is_correlated"]

    stages = [stage["stage"] for stage in state["correlation"]["attack_chain"]]
    assert stages == ["Reconnaissance", "Credential Access", "Privilege Escalation"]
    assert state["risk_assessment"]["risk_level"] == "CRITICAL"


def test_email_routes_as_email(workflow):
    run = workflow.analyze(PHISHING_EMAIL)
    assert run.state["input_type"] == "email"
    assert run.state["threat_analysis"]["attack_type"] == AttackType.PHISHING.value


def test_trace_records_every_executed_node(workflow):
    run = workflow.analyze(BRUTE_FORCE_EVENT)
    path = node_path(run.state)

    assert path[:2] == ["input_classifier", "threat_detector"]
    assert "risk_assessment" in path
    assert all(entry["status"] == "success" for entry in run.state["node_trace"])
    assert all(entry["latency_seconds"] >= 0 for entry in run.state["node_trace"])


def test_node_failure_is_captured_not_raised(backend, retriever, monkeypatch):
    """A failing agent must degrade the run, not crash the workflow."""
    import cybersentinel.graph.nodes as nodes_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated retrieval outage")

    monkeypatch.setattr(nodes_module, "gather_intelligence", boom)

    from cybersentinel.graph.workflow import CyberSentinelWorkflow

    failing = CyberSentinelWorkflow(backend=backend, retriever=retriever, enable_interrupt=False)
    run = failing.analyze(BRUTE_FORCE_EVENT)

    assert any("simulated retrieval outage" in error for error in run.errors)
    assert run.report, "the workflow must still produce a report"
    assert run.report["risk"]["risk_score"] > 0


def test_empty_input_is_rejected(workflow):
    from cybersentinel.utils.validation import InputValidationError

    with pytest.raises(InputValidationError):
        workflow.analyze("   ")


def test_repeated_runs_use_separate_threads(workflow):
    first = workflow.analyze(BRUTE_FORCE_EVENT)
    second = workflow.analyze(BRUTE_FORCE_EVENT)

    assert first.incident_id != second.incident_id
    assert first.thread_id != second.thread_id


def test_graph_renders_a_diagram(workflow):
    diagram = workflow.mermaid()
    assert "human_approval" in diagram
    assert "incident_report" in diagram
