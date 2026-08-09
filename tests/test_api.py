"""API and persistence tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cybersentinel.api.main import create_app
from cybersentinel.api.routes.incidents import get_retriever
from cybersentinel.database.repository import IncidentRepository
from cybersentinel.service import get_service
from tests.conftest import BENIGN_EVENT, BRUTE_FORCE_EVENT, MULTI_EVENT


@pytest.fixture
def client(service, retriever) -> TestClient:
    """A test client whose dependencies point at the temporary service."""
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_retriever] = lambda: retriever
    return TestClient(app)


# --------------------------------------------------------------------------- #
# System endpoints
# --------------------------------------------------------------------------- #
def test_health_reports_every_component(client):
    payload = client.get("/health").json()

    assert payload["status"] in ("ok", "degraded")
    names = {component["name"] for component in payload["components"]}
    assert names == {"llm", "rag", "database"}


def test_root_lists_entry_points(client):
    payload = client.get("/").json()
    assert payload["name"] == "CyberSentinel"
    assert payload["docs"] == "/docs"


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert "/analyze" in schema["paths"]
    assert "/approval/{incident_id}" in schema["paths"]


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def test_analyze_high_risk_pauses_for_approval(client):
    response = client.post("/analyze", json={"text": BRUTE_FORCE_EVENT})
    payload = response.json()

    assert response.status_code == 200
    assert payload["awaiting_approval"]
    assert payload["report"] is None
    assert payload["pending_approval"]["risk_level"] in ("HIGH", "CRITICAL")
    assert payload["pending_approval"]["high_impact_actions"]
    assert payload["node_path"][0] == "input_classifier"


def test_analyze_benign_completes_immediately(client):
    payload = client.post("/analyze", json={"text": BENIGN_EVENT}).json()

    assert not payload["awaiting_approval"]
    assert payload["report"]["attack_type"] == "Benign"
    assert payload["report"]["risk"]["risk_level"] == "LOW"
    assert payload["pending_approval"] is None


def test_analyze_returns_grounded_intelligence(client):
    payload = client.post("/analyze", json={"text": BRUTE_FORCE_EVENT}).json()
    techniques = payload["pending_approval"]["mitre_techniques"]

    assert techniques
    assert all(technique.startswith("T") for technique in techniques)


def test_analyze_without_rag_reports_no_sources(client):
    payload = client.post("/analyze", json={"text": BRUTE_FORCE_EVENT, "use_rag": False}).json()
    assert payload["pending_approval"]["mitre_techniques"] == []


def test_analyze_multi_event_correlates(client):
    payload = client.post("/analyze", json={"text": MULTI_EVENT}).json()
    assert payload["awaiting_approval"]
    assert payload["pending_approval"]["risk_level"] == "CRITICAL"


def test_batch_analysis(client):
    response = client.post(
        "/analyze/batch", json={"events": [BRUTE_FORCE_EVENT, BENIGN_EVENT]}
    )
    payload = response.json()

    assert payload["submitted"] == 2
    assert payload["analysed"] == 2
    assert len({result["incident_id"] for result in payload["results"]}) == 2


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #
def test_approval_flow(client):
    started = client.post("/analyze", json={"text": BRUTE_FORCE_EVENT}).json()

    response = client.post(
        f"/approval/{started['thread_id']}",
        json={"decision": "APPROVED", "decided_by": "analyst", "note": "confirmed"},
    )
    payload = response.json()

    assert response.status_code == 200
    assert not payload["awaiting_approval"]
    assert payload["report"]["approval"]["decision"] == "APPROVED"
    assert payload["report"]["approval"]["decided_by"] == "analyst"


def test_rejection_removes_disruptive_recommendations(client):
    started = client.post("/analyze", json={"text": BRUTE_FORCE_EVENT}).json()
    payload = client.post(
        f"/approval/{started['thread_id']}", json={"decision": "REJECTED"}
    ).json()

    assert payload["report"]["approval"]["decision"] == "REJECTED"
    assert not any(item["high_impact"] for item in payload["report"]["recommendations"])


def test_approval_for_unknown_incident_conflicts(client):
    response = client.post("/approval/INC-DOES-NOT-EXIST", json={"decision": "APPROVED"})
    assert response.status_code == 409


def test_invalid_decision_is_rejected(client):
    started = client.post("/analyze", json={"text": BRUTE_FORCE_EVENT}).json()
    response = client.post(
        f"/approval/{started['thread_id']}", json={"decision": "EXECUTE_NOW"}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("body", [{"text": "   "}, {"text": ""}, {}])
def test_invalid_payloads_are_rejected(client, body):
    assert client.post("/analyze", json=body).status_code == 422


def test_oversized_input_is_rejected(client):
    response = client.post("/analyze", json={"text": "a" * 20_001})
    assert response.status_code == 422


def test_validation_error_response_is_serialisable(client):
    payload = client.post("/analyze", json={"text": "   "}).json()
    assert payload["detail"] == "invalid request payload"
    assert isinstance(payload["errors"], list)
    assert "message" in payload["errors"][0]


def test_asset_criticality_is_bounded(client):
    assert client.post(
        "/analyze", json={"text": BENIGN_EVENT, "asset_criticality": 9}
    ).status_code == 422


# --------------------------------------------------------------------------- #
# Incident history
# --------------------------------------------------------------------------- #
def test_incidents_are_listed_after_analysis(client):
    client.post("/analyze", json={"text": BRUTE_FORCE_EVENT})
    rows = client.get("/incidents").json()

    assert rows
    assert rows[0]["attack_type"] == "Brute Force"
    assert "incident_id" in rows[0]


def test_incident_detail_and_missing_incident(client):
    started = client.post("/analyze", json={"text": BRUTE_FORCE_EVENT}).json()

    found = client.get(f"/incidents/{started['incident_id']}")
    assert found.status_code == 200
    assert found.json()["incident_id"] == started["incident_id"]

    assert client.get("/incidents/INC-NOT-REAL").status_code == 404


def test_pending_approvals_endpoint(client):
    client.post("/analyze", json={"text": BRUTE_FORCE_EVENT})
    pending = client.get("/incidents/pending-approval").json()

    assert pending
    assert all(row["approval_status"] == "PENDING" for row in pending)


def test_indicator_search_finds_previous_incidents(client):
    client.post("/analyze", json={"text": BRUTE_FORCE_EVENT})
    rows = client.get("/indicators/search", params={"value": "198.51.100.23"}).json()

    assert rows
    assert rows[0]["attack_type"] == "Brute Force"


def test_history_matches_are_returned_on_repeat_indicator(client):
    client.post("/analyze", json={"text": BRUTE_FORCE_EVENT})
    payload = client.post("/analyze", json={"text": BRUTE_FORCE_EVENT}).json()

    kinds = {match["kind"] for match in payload["history_matches"]}
    assert "ips" in kinds


def test_metrics_endpoint(client):
    client.post("/analyze", json={"text": BRUTE_FORCE_EVENT})
    client.post("/analyze", json={"text": BENIGN_EVENT})
    metrics = client.get("/metrics").json()

    assert metrics["total_incidents"] >= 2
    assert metrics["by_attack_type"]["Brute Force"] >= 1
    assert metrics["pending_approvals"] >= 1


# --------------------------------------------------------------------------- #
# Threat-intelligence search
# --------------------------------------------------------------------------- #
def test_threat_intelligence_search(client):
    payload = client.get(
        "/threat-intelligence/search", params={"q": "brute force T1110"}
    ).json()

    assert payload["documents"]
    assert any(
        str(document["document_id"]).startswith("T1110") for document in payload["documents"]
    )
    assert all(document["source"] for document in payload["documents"])


def test_search_query_must_be_long_enough(client):
    assert client.get("/threat-intelligence/search", params={"q": "a"}).status_code == 422


# --------------------------------------------------------------------------- #
# Repository
# --------------------------------------------------------------------------- #
def test_repository_persists_and_updates(service, session_factory):
    result = service.analyze(BRUTE_FORCE_EVENT)

    with session_factory() as session:
        repository = IncidentRepository(session)
        incident = repository.get(result.incident_id)

        assert incident is not None
        assert incident.attack_type == "Brute Force"
        assert incident.approval_status == "PENDING"
        assert {indicator.value for indicator in incident.indicators} >= {"198.51.100.23"}

    service.submit_decision(result.thread_id, "APPROVED", decided_by="analyst")

    with session_factory() as session:
        incident = IncidentRepository(session).get(result.incident_id)
        assert incident.approval_status == "APPROVED"
        assert len(incident.approvals) == 1
        assert incident.approvals[0].decided_by == "analyst"


def test_repository_input_preview_is_redacted(service, session_factory):
    service.analyze(
        "Login failure for user@example.com with password=hunter2 from 198.51.100.23", persist=True
    )

    with session_factory() as session:
        incident = IncidentRepository(session).list_incidents(limit=1)[0]
        assert "hunter2" not in incident.input_preview
        assert "user@example.com" not in incident.input_preview


def test_repository_metrics_and_similarity(service, session_factory):
    service.analyze(BRUTE_FORCE_EVENT)
    service.analyze(BENIGN_EVENT)

    with session_factory() as session:
        repository = IncidentRepository(session)
        metrics = repository.metrics()

        assert metrics["total_incidents"] == 2
        assert repository.find_similar("Brute Force")
        assert repository.count() == 2
