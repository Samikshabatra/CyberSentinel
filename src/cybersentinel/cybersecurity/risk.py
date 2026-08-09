"""Deterministic risk model.

Risk is computed in plain Python, not by the LLM, so the score is reproducible
and fully explainable during review. This is a *project* risk model, not a
universal cybersecurity standard - see docs/security.md.

    Risk Score = Likelihood (1-5) x Impact (1-5)

    1-4   -> LOW
    5-9   -> MEDIUM
    10-16 -> HIGH
    17-25 -> CRITICAL
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cybersentinel.cybersecurity.taxonomy import AttackType, Severity

LIKELIHOOD_SCALE: dict[int, str] = {
    1: "Rare",
    2: "Unlikely",
    3: "Possible",
    4: "Likely",
    5: "Almost Certain",
}

IMPACT_SCALE: dict[int, str] = {
    1: "Negligible",
    2: "Minor",
    3: "Moderate",
    4: "Major",
    5: "Severe",
}

RISK_BANDS: tuple[tuple[int, int, Severity], ...] = (
    (1, 4, Severity.LOW),
    (5, 9, Severity.MEDIUM),
    (10, 16, Severity.HIGH),
    (17, 25, Severity.CRITICAL),
)

#: Baseline impact per attack type - how much damage the category can do if real.
BASE_IMPACT: dict[AttackType, int] = {
    AttackType.PHISHING: 3,
    AttackType.BRUTE_FORCE: 3,
    AttackType.CREDENTIAL_ATTACK: 4,
    AttackType.SQL_INJECTION: 4,
    AttackType.XSS: 3,
    AttackType.MALWARE: 5,
    AttackType.DDOS: 4,
    AttackType.RECONNAISSANCE: 2,
    AttackType.PRIVILEGE_ESCALATION: 5,
    AttackType.DATA_EXFILTRATION: 5,
    AttackType.SUSPICIOUS_AUTH: 3,
    AttackType.VULNERABILITY: 4,
    AttackType.INSIDER_THREAT: 4,
    AttackType.BENIGN: 1,
    AttackType.UNKNOWN: 2,
}

#: Actions considered operationally impactful; these always need analyst approval.
HIGH_IMPACT_ACTION_MARKERS: tuple[str, ...] = (
    "block",
    "isolate",
    "quarantine",
    "disable account",
    "lock account",
    "suspend",
    "shut down",
    "shutdown",
    "revoke",
    "terminate",
    "reimage",
    "firewall rule",
    "takedown",
)


@dataclass(frozen=True)
class RiskAssessment:
    """Transparent risk result: inputs, score, band and the reasons behind it."""

    likelihood: int
    impact: int
    risk_score: int
    risk_level: Severity
    likelihood_label: str
    impact_label: str
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "likelihood": self.likelihood,
            "impact": self.impact,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "likelihood_label": self.likelihood_label,
            "impact_label": self.impact_label,
            "rationale": list(self.rationale),
            "formula": "risk_score = likelihood x impact",
        }


def _clamp(value: int, low: int = 1, high: int = 5) -> int:
    return max(low, min(high, value))


def score_to_level(score: int) -> Severity:
    """Map a 1-25 risk score onto a severity band."""
    for low, high, level in RISK_BANDS:
        if low <= score <= high:
            return level
    return Severity.UNKNOWN


def derive_likelihood(
    confidence: float,
    evidence_count: int,
    corroborating_events: int = 0,
) -> tuple[int, list[str]]:
    """Derive a 1-5 likelihood from model confidence and evidence volume.

    Confidence sets the base; independent evidence items and corroborating
    correlated events raise it. Every adjustment is recorded for explainability.
    """
    confidence = max(0.0, min(1.0, confidence))
    if confidence >= 0.90:
        likelihood, band = 5, "confidence >= 0.90"
    elif confidence >= 0.75:
        likelihood, band = 4, "confidence 0.75-0.89"
    elif confidence >= 0.55:
        likelihood, band = 3, "confidence 0.55-0.74"
    elif confidence >= 0.35:
        likelihood, band = 2, "confidence 0.35-0.54"
    else:
        likelihood, band = 1, "confidence < 0.35"

    rationale = [f"Base likelihood {likelihood} from detection {band}."]

    if evidence_count == 0:
        likelihood -= 1
        rationale.append("No supporting evidence extracted: likelihood reduced by 1.")
    elif evidence_count >= 3:
        likelihood += 1
        rationale.append(f"{evidence_count} independent evidence items: likelihood raised by 1.")

    if corroborating_events >= 2:
        likelihood += 1
        rationale.append(
            f"{corroborating_events} correlated events support the finding: likelihood raised by 1."
        )

    return _clamp(likelihood), rationale


def derive_impact(
    attack_type: AttackType,
    detected_severity: Severity = Severity.UNKNOWN,
    asset_criticality: int | None = None,
) -> tuple[int, list[str]]:
    """Derive a 1-5 impact from the attack category, model severity and asset value."""
    impact = BASE_IMPACT.get(attack_type, 2)
    rationale = [f"Base impact {impact} for category '{attack_type.value}'."]

    if detected_severity is Severity.CRITICAL:
        impact += 1
        rationale.append("Model assessed severity CRITICAL: impact raised by 1.")
    elif detected_severity is Severity.LOW:
        impact -= 1
        rationale.append("Model assessed severity LOW: impact reduced by 1.")

    if asset_criticality is not None:
        criticality = _clamp(asset_criticality)
        if criticality >= 4:
            impact += 1
            rationale.append(f"High asset criticality ({criticality}/5): impact raised by 1.")
        elif criticality <= 2:
            impact -= 1
            rationale.append(f"Low asset criticality ({criticality}/5): impact reduced by 1.")

    return _clamp(impact), rationale


def assess_risk(
    attack_type: AttackType,
    confidence: float,
    evidence_count: int,
    detected_severity: Severity = Severity.UNKNOWN,
    corroborating_events: int = 0,
    asset_criticality: int | None = None,
) -> RiskAssessment:
    """Compute a full, explainable risk assessment."""
    if attack_type is AttackType.BENIGN:
        # Confidence here is confidence that the activity is *normal*, so it must
        # not be fed into likelihood-of-threat. A confidently benign event is the
        # lowest risk there is, not a likely one.
        return RiskAssessment(
            likelihood=1,
            impact=1,
            risk_score=1,
            risk_level=Severity.LOW,
            likelihood_label=LIKELIHOOD_SCALE[1],
            impact_label=IMPACT_SCALE[1],
            rationale=[
                f"Activity classified as benign with confidence {confidence:.2f}.",
                "Likelihood and impact are both set to 1: no threat is indicated.",
                "Risk score = 1 x 1 = 1 -> LOW.",
            ],
        )

    likelihood, likelihood_reasons = derive_likelihood(
        confidence, evidence_count, corroborating_events
    )
    impact, impact_reasons = derive_impact(attack_type, detected_severity, asset_criticality)

    score = likelihood * impact
    level = score_to_level(score)

    rationale = [
        *likelihood_reasons,
        *impact_reasons,
        f"Risk score = {likelihood} x {impact} = {score} -> {level.value}.",
    ]

    if attack_type is AttackType.UNKNOWN:
        rationale.append(
            "Category is Unknown: treat this assessment as provisional pending analyst review."
        )

    return RiskAssessment(
        likelihood=likelihood,
        impact=impact,
        risk_score=score,
        risk_level=level,
        likelihood_label=LIKELIHOOD_SCALE[likelihood],
        impact_label=IMPACT_SCALE[impact],
        rationale=rationale,
    )


def is_high_impact_action(action: str) -> bool:
    """True when a recommendation would materially change production state."""
    lowered = action.lower()
    return any(marker in lowered for marker in HIGH_IMPACT_ACTION_MARKERS)


def requires_human_approval(
    risk_level: Severity,
    recommendations: list[str] | None = None,
    threshold: Severity = Severity.HIGH,
) -> tuple[bool, str]:
    """Decide whether analyst approval is needed, with the triggering reason."""
    from cybersentinel.cybersecurity.taxonomy import severity_at_least

    if severity_at_least(risk_level, threshold):
        return True, f"Risk level {risk_level.value} is at or above the {threshold.value} threshold."

    for action in recommendations or []:
        if is_high_impact_action(action):
            return True, f"Recommendation is operationally impactful: '{action}'."

    return False, "Risk level and recommendations are below the approval threshold."


def risk_matrix() -> list[list[int]]:
    """Return the 5x5 likelihood-by-impact score matrix (for UI/docs)."""
    return [[likelihood * impact for impact in range(1, 6)] for likelihood in range(1, 6)]
