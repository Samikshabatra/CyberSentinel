"""Agent-level tests: classification, detection, risk, correlation, response, report."""

from __future__ import annotations

import pytest

from cybersentinel.agents.correlation import correlate_events
from cybersentinel.agents.input_classifier import classify_input
from cybersentinel.agents.report import build_report
from cybersentinel.agents.response import is_prohibited, recommend
from cybersentinel.agents.risk_assessment import assess
from cybersentinel.agents.threat_detector import aggregate_detections, detect_batch, detect_threat
from cybersentinel.agents.threat_intelligence import gather_intelligence
from cybersentinel.cybersecurity.risk import (
    assess_risk,
    is_high_impact_action,
    requires_human_approval,
    score_to_level,
)
from cybersentinel.cybersecurity.taxonomy import (
    ApprovalDecision,
    AttackType,
    InputType,
    Severity,
    normalise_attack_type,
    normalise_severity,
    severity_at_least,
)
from cybersentinel.schemas.analysis import (
    ApprovalRecord,
    CorrelationResult,
    MitreMapping,
    ThreatAnalysis,
)
from tests.conftest import (
    BENIGN_EVENT,
    BRUTE_FORCE_EVENT,
    MULTI_EVENT,
    PHISHING_EMAIL,
    VAGUE_EVENT,
)


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("brute force", AttackType.BRUTE_FORCE),
        ("BRUTE_FORCE", AttackType.BRUTE_FORCE),
        ("sqli", AttackType.SQL_INJECTION),
        ("Spear Phishing", AttackType.PHISHING),
        ("ransomware", AttackType.MALWARE),
        ("something invented", AttackType.UNKNOWN),
        (None, AttackType.UNKNOWN),
        ("", AttackType.UNKNOWN),
    ],
)
def test_attack_type_normalisation(raw, expected):
    assert normalise_attack_type(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("high", Severity.HIGH), ("Critical", Severity.CRITICAL), ("nonsense", Severity.UNKNOWN)],
)
def test_severity_normalisation(raw, expected):
    assert normalise_severity(raw) is expected


def test_severity_ordering():
    assert severity_at_least(Severity.CRITICAL, Severity.HIGH)
    assert severity_at_least(Severity.HIGH, Severity.HIGH)
    assert not severity_at_least(Severity.MEDIUM, Severity.HIGH)


# --------------------------------------------------------------------------- #
# Input classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (BRUTE_FORCE_EVENT, InputType.ALERT),
        (PHISHING_EMAIL, InputType.EMAIL),
        ("https://login.example.net/verify?id=99", InputType.URL),
        (MULTI_EVENT, InputType.MULTI_EVENT),
        (
            "Scan reports db-prod-02 affected by CVE-2024-12345 with a CVSS score of 9.8.",
            InputType.VULNERABILITY,
        ),
    ],
)
def test_input_classification(text, expected):
    assert classify_input(text).input_type is expected


def test_email_is_not_split_into_multiple_events():
    """An email's blank line separates headers from body, not two events."""
    classification = classify_input(PHISHING_EMAIL)
    assert classification.input_type is InputType.EMAIL
    assert classification.event_count == 1


def test_multi_event_counts_events():
    classification = classify_input(MULTI_EVENT)
    assert classification.event_count == 4


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def test_detects_brute_force(backend):
    outcome = detect_threat(BRUTE_FORCE_EVENT, backend=backend)
    assert outcome.analysis.attack_type is AttackType.BRUTE_FORCE
    assert outcome.analysis.evidence
    assert outcome.valid_json
    assert outcome.error is None


def test_detects_benign(backend):
    outcome = detect_threat(BENIGN_EVENT, backend=backend)
    assert outcome.analysis.attack_type is AttackType.BENIGN


def test_returns_unknown_without_evidence(backend):
    """The system must be able to decline rather than guess."""
    outcome = detect_threat(VAGUE_EVENT, backend=backend)
    assert outcome.analysis.attack_type is AttackType.UNKNOWN
    assert outcome.analysis.evidence == []


def test_unknown_classification_cannot_claim_high_severity():
    analysis = ThreatAnalysis(attack_type=AttackType.UNKNOWN, severity=Severity.CRITICAL)
    assert analysis.severity is Severity.UNKNOWN


def test_aggregation_selects_most_severe_conclusive_detection(backend):
    events = [
        "Port scan from 203.0.113.45 against 1200 sequential ports",
        "User admin added to the administrators group shortly after login",
    ]
    outcomes = detect_batch(events, backend=backend)
    aggregated = aggregate_detections(outcomes)

    assert aggregated.attack_type is AttackType.PRIVILEGE_ESCALATION
    assert len(aggregated.evidence) >= 2


def test_aggregation_of_inconclusive_events_stays_unknown(backend):
    outcomes = detect_batch(["Alert.", "Investigate."], backend=backend)
    assert aggregate_detections(outcomes).attack_type is AttackType.UNKNOWN


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
def test_risk_score_is_likelihood_times_impact():
    assessment = assess_risk(AttackType.BRUTE_FORCE, confidence=0.94, evidence_count=3)
    assert assessment.risk_score == assessment.likelihood * assessment.impact
    assert assessment.rationale


@pytest.mark.parametrize(
    ("score", "expected"),
    [(1, Severity.LOW), (4, Severity.LOW), (5, Severity.MEDIUM), (9, Severity.MEDIUM),
     (10, Severity.HIGH), (16, Severity.HIGH), (17, Severity.CRITICAL), (25, Severity.CRITICAL)],
)
def test_risk_band_boundaries(score, expected):
    assert score_to_level(score) is expected


def test_low_confidence_and_no_evidence_lowers_likelihood():
    weak = assess_risk(AttackType.BRUTE_FORCE, confidence=0.2, evidence_count=0)
    strong = assess_risk(AttackType.BRUTE_FORCE, confidence=0.95, evidence_count=4)
    assert weak.likelihood < strong.likelihood
    assert weak.risk_score < strong.risk_score


def test_correlated_events_raise_likelihood():
    alone = assess_risk(AttackType.PRIVILEGE_ESCALATION, 0.8, 2)
    corroborated = assess_risk(AttackType.PRIVILEGE_ESCALATION, 0.8, 2, corroborating_events=3)
    assert corroborated.likelihood >= alone.likelihood


def test_risk_assessment_model_rejects_inconsistent_score():
    from pydantic import ValidationError

    from cybersentinel.schemas.analysis import RiskAssessmentModel

    with pytest.raises(ValidationError):
        RiskAssessmentModel(
            likelihood=3,
            impact=3,
            risk_score=99,
            risk_level=Severity.HIGH,
            likelihood_label="Possible",
            impact_label="Moderate",
        )


# --------------------------------------------------------------------------- #
# Approval policy
# --------------------------------------------------------------------------- #
def test_high_risk_requires_approval():
    required, reason = requires_human_approval(Severity.HIGH, [])
    assert required
    assert "HIGH" in reason


def test_low_risk_with_disruptive_action_still_requires_approval():
    required, reason = requires_human_approval(Severity.LOW, ["Block the source IP"])
    assert required
    assert "impactful" in reason


def test_low_risk_investigation_needs_no_approval():
    required, _ = requires_human_approval(Severity.LOW, ["Review authentication logs"])
    assert not required


@pytest.mark.parametrize(
    "action",
    ["Block the source IP", "Isolate the affected host", "Revoke active sessions",
     "Disable account for the user"],
)
def test_disruptive_actions_are_flagged(action):
    assert is_high_impact_action(action)


@pytest.mark.parametrize(
    "action", ["Review authentication logs", "Preserve forensic evidence", "Enable MFA"]
)
def test_investigative_actions_are_not_flagged(action):
    assert not is_high_impact_action(action)


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
def test_correlation_builds_ordered_chain(backend):
    events = [
        "Port scan from 203.0.113.45 against 1200 sequential ports",
        "20 failed SSH logins for user admin from 203.0.113.45",
        "User admin added to the administrators group from 203.0.113.45",
    ]
    types = [
        AttackType.RECONNAISSANCE,
        AttackType.BRUTE_FORCE,
        AttackType.PRIVILEGE_ESCALATION,
    ]
    result = correlate_events(events, types, backend=backend, use_llm=False)

    assert result.is_correlated
    stages = [stage.stage for stage in result.attack_chain]
    assert stages == ["Reconnaissance", "Credential Access", "Privilege Escalation"]
    assert "203.0.113.45" in result.shared_indicators.get("ips", [])


def test_unrelated_events_are_not_correlated(backend):
    events = [
        "Port scan from 203.0.113.45 against many ports",
        "20 failed SSH logins for user admin from 198.51.100.99",
    ]
    result = correlate_events(
        events, [AttackType.RECONNAISSANCE, AttackType.BRUTE_FORCE], backend=backend, use_llm=False
    )
    assert not result.is_correlated
    assert result.attack_chain == []


def test_single_event_is_not_correlated(backend):
    result = correlate_events(
        [BRUTE_FORCE_EVENT], [AttackType.BRUTE_FORCE], backend=backend, use_llm=False
    )
    assert not result.is_correlated
    assert result.confidence == 0.0


# --------------------------------------------------------------------------- #
# Threat intelligence grounding
# --------------------------------------------------------------------------- #
def test_grounded_mapping_uses_retrieved_sources(backend, retriever):
    analysis = ThreatAnalysis(
        attack_type=AttackType.BRUTE_FORCE,
        severity=Severity.HIGH,
        confidence=0.9,
        evidence=["47 failed authentication attempts"],
        candidate_techniques=["T1110"],
    )
    outcome = gather_intelligence(analysis, BRUTE_FORCE_EVENT, retriever=retriever, backend=backend)

    assert outcome.mapping.grounded
    assert any(t["technique_id"].startswith("T1110") for t in outcome.mapping.techniques)
    assert outcome.sources


def test_invented_technique_is_rejected(backend, retriever):
    analysis = ThreatAnalysis(
        attack_type=AttackType.BRUTE_FORCE,
        severity=Severity.HIGH,
        confidence=0.9,
        evidence=["47 failed authentication attempts"],
        candidate_techniques=["T1110", "T9999"],
    )
    outcome = gather_intelligence(analysis, BRUTE_FORCE_EVENT, retriever=retriever, backend=backend)

    reported = {t["technique_id"] for t in outcome.mapping.techniques}
    assert "T9999" not in reported
    assert "T9999" in outcome.mapping.rejected_claims


def test_benign_event_gets_no_threat_intelligence(backend, retriever):
    analysis = ThreatAnalysis(attack_type=AttackType.BENIGN, severity=Severity.LOW, confidence=0.9)
    outcome = gather_intelligence(analysis, BENIGN_EVENT, retriever=retriever, backend=backend)

    assert outcome.mapping.techniques == []
    assert not outcome.mapping.grounded


def test_empty_store_falls_back_to_verified_catalogue(backend, empty_retriever):
    """Retrieval failure must not invent identifiers, and must not claim grounding."""
    analysis = ThreatAnalysis(
        attack_type=AttackType.BRUTE_FORCE,
        severity=Severity.HIGH,
        confidence=0.9,
        evidence=["failures"],
    )
    outcome = gather_intelligence(
        analysis, BRUTE_FORCE_EVENT, retriever=empty_retriever, backend=backend
    )

    assert outcome.used_catalogue_fallback
    assert not outcome.mapping.grounded
    from cybersentinel.cybersecurity.mitre import TECHNIQUES

    for technique in outcome.mapping.techniques:
        assert technique["technique_id"] in TECHNIQUES


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #
def _risk(level: Severity = Severity.HIGH):
    return assess(
        ThreatAnalysis(
            attack_type=AttackType.BRUTE_FORCE,
            severity=level,
            confidence=0.9,
            evidence=["failures", "same source", "short window"],
        )
    )


def test_recommendations_are_produced_and_flagged(backend):
    analysis = ThreatAnalysis(
        attack_type=AttackType.BRUTE_FORCE,
        severity=Severity.HIGH,
        confidence=0.9,
        evidence=["47 failed attempts"],
    )
    recommendations = recommend(analysis, _risk(), MitreMapping(), backend=backend)

    assert recommendations
    assert any(item.high_impact for item in recommendations)
    assert all(item.requires_approval == item.high_impact for item in recommendations)


def test_investigative_actions_are_ordered_before_disruptive_ones(backend):
    analysis = ThreatAnalysis(
        attack_type=AttackType.MALWARE,
        severity=Severity.CRITICAL,
        confidence=0.9,
        evidence=["mass file modification"],
    )
    recommendations = recommend(analysis, _risk(Severity.CRITICAL), MitreMapping(), backend=backend)
    flags = [item.high_impact for item in recommendations]
    assert flags == sorted(flags)


@pytest.mark.parametrize(
    "action",
    [
        "Hack back the attacker infrastructure",
        "Launch a penetration test against the source",
        "Scan the attacker network for weaknesses",
        "Delete all the logs to clean up",
        "Disable logging on the affected host",
        "Pay the ransom to recover the files",
    ],
)
def test_offensive_and_destructive_actions_are_prohibited(action):
    assert is_prohibited(action)


@pytest.mark.parametrize(
    "action",
    [
        "Review authentication logs for the source IP",
        "Enable multi-factor authentication",
        "Isolate the affected host from the network",
        "Preserve forensic evidence before containment",
    ],
)
def test_defensive_actions_are_allowed(action):
    assert not is_prohibited(action)


def test_prohibited_model_output_is_filtered_out(backend, monkeypatch):
    """Even if the model proposes an offensive action, it must not reach the analyst."""
    import cybersentinel.agents.response as response_module

    def fake_generate(*_args, **_kwargs):
        from cybersentinel.llm.inference import GenerationResult

        return GenerationResult(
            text=(
                '{"recommendations": ['
                '{"action": "Hack back the attacker", "priority": "HIGH"},'
                '{"action": "Review authentication logs", "priority": "HIGH"}]}'
            ),
            latency_seconds=0.0,
            backend="test",
        )

    monkeypatch.setattr(response_module, "generate", fake_generate)

    analysis = ThreatAnalysis(
        attack_type=AttackType.BRUTE_FORCE, severity=Severity.HIGH, confidence=0.9, evidence=["x"]
    )
    actions = [item.action for item in recommend(analysis, _risk(), MitreMapping(), backend=backend)]

    assert "Hack back the attacker" not in actions
    assert "Review authentication logs" in actions


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def test_report_answers_the_explainability_questions(backend):
    analysis = ThreatAnalysis(
        attack_type=AttackType.BRUTE_FORCE,
        severity=Severity.HIGH,
        confidence=0.9,
        evidence=["47 failed attempts"],
        reasoning="Repeated failures from one source.",
    )
    report = build_report(
        incident_id="INC-TEST-1",
        run_id="run1",
        input_type=InputType.ALERT,
        analysis=analysis,
        mapping=MitreMapping(),
        correlation=CorrelationResult(),
        risk=_risk(),
        recommendations=recommend(analysis, _risk(), MitreMapping(), backend=backend),
        approval=ApprovalRecord(decision=ApprovalDecision.APPROVED, required=True),
        sources=[],
        backend=backend,
    )

    for key in (
        "what_was_detected",
        "why_detected",
        "evidence",
        "confidence",
        "threat_intelligence_sources",
        "mitre_techniques",
        "risk",
        "next_steps",
    ):
        assert report.explainability[key]

    assert report.disclaimer
    assert "require" in report.disclaimer.lower()


def test_report_for_unknown_does_not_assert_a_threat(backend):
    analysis = ThreatAnalysis(attack_type=AttackType.UNKNOWN, confidence=0.1)
    report = build_report(
        incident_id="INC-TEST-2",
        run_id="run2",
        input_type=InputType.ALERT,
        analysis=analysis,
        mapping=MitreMapping(),
        correlation=CorrelationResult(),
        risk=None,
        recommendations=[],
        approval=ApprovalRecord(),
        sources=[],
        backend=backend,
    )

    assert report.is_insufficient
    assert "not contain sufficient evidence" in report.summary
    assert report.mitre.techniques == []
