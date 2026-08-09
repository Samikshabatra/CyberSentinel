"""Response recommendation agent (LangGraph node 7).

Produces defensive recommendations only. Two safety properties are enforced in
code rather than trusted to the prompt:

* recommendations matching offensive or destructive verbs are dropped,
* any operationally impactful action is flagged so approval routing can catch it.

If the model fails or every suggestion is filtered out, a deterministic
playbook for the category is used - the analyst is never left with nothing.
"""

from __future__ import annotations

import re

from cybersentinel.cybersecurity.risk import is_high_impact_action
from cybersentinel.cybersecurity.taxonomy import Severity, normalise_severity
from cybersentinel.llm.inference import generate
from cybersentinel.llm.model import LLMBackend, fallback_playbook
from cybersentinel.llm.prompts import build_response_messages
from cybersentinel.llm.structured_output import parse_json_object
from cybersentinel.schemas.analysis import (
    MitreMapping,
    Recommendation,
    RiskAssessmentModel,
    ThreatAnalysis,
)
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)

#: Actions the system must never recommend. This is a defensive analysis tool.
_PROHIBITED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:hack|attack|exploit|pwn|compromise)\s+(?:back|the attacker|their)",
        r"\bcounter[- ]?(?:attack|hack)\b",
        r"\b(?:launch|run|perform|execute)\s+(?:a\s+)?(?:penetration test|pentest|port scan|nmap|exploit)",
        r"\bscan\s+the\s+(?:attacker|source|external)\b",
        r"\bdelete\s+(?:all\s+)?(?:the\s+)?logs\b",
        r"\b(?:clear|wipe|purge)\s+(?:all\s+)?(?:the\s+)?(?:event\s+)?logs\b",
        r"\bdisable\s+(?:logging|auditing|monitoring)\b",
        r"\bpay\s+the\s+ransom\b",
        r"\bformat\s+the\s+(?:disk|drive)\b",
        r"\bwipe\s+the\s+(?:server|system|host)\b",
    )
)


def is_prohibited(action: str) -> bool:
    """True when an action is offensive, destructive or evidence-destroying."""
    return any(pattern.search(action) for pattern in _PROHIBITED_PATTERNS)


def _to_recommendation(item: dict[str, object], default_priority: Severity) -> Recommendation | None:
    action = str(item.get("action", "")).strip()
    if not action or is_prohibited(action):
        if action:
            logger.warning(f"dropped prohibited recommendation: {action!r}")
        return None

    priority = normalise_severity(str(item.get("priority", "")))
    if priority is Severity.UNKNOWN:
        priority = default_priority

    high_impact = bool(item.get("high_impact", False)) or is_high_impact_action(action)

    return Recommendation(
        action=action,
        rationale=str(item.get("rationale", "")).strip(),
        priority=priority,
        high_impact=high_impact,
        requires_approval=high_impact,
    )


def recommend(
    analysis: ThreatAnalysis,
    risk: RiskAssessmentModel,
    mapping: MitreMapping | None = None,
    backend: LLMBackend | None = None,
    use_llm: bool = True,
) -> list[Recommendation]:
    """Generate validated defensive recommendations for an incident."""
    default_priority = risk.risk_level if risk.risk_level is not Severity.UNKNOWN else Severity.MEDIUM
    recommendations: list[Recommendation] = []

    if use_llm:
        intel_text = ""
        if mapping:
            intel_text = ", ".join(
                technique.get("technique_id", "") for technique in mapping.techniques
            )
        messages = build_response_messages(
            analysis.attack_type.value,
            analysis.severity.value,
            risk.risk_level.value,
            risk.risk_score,
            analysis.evidence,
            intel_text,
        )
        result = generate(messages, backend=backend)
        if result.ok:
            parsed = parse_json_object(result.text)
            items = (parsed.data or {}).get("recommendations", []) if parsed.ok else []
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    recommendation = _to_recommendation(item, default_priority)
                    if recommendation:
                        recommendations.append(recommendation)
        else:
            logger.warning(f"response model call failed: {result.error}")

    if not recommendations:
        logger.info("using deterministic playbook for response recommendations")
        for item in fallback_playbook(analysis.attack_type.value):
            recommendation = _to_recommendation(item, default_priority)
            if recommendation:
                recommendations.append(recommendation)

    # Investigation before disruption: non-impactful actions come first.
    order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3, Severity.UNKNOWN: 4}
    recommendations.sort(key=lambda item: (item.high_impact, order[item.priority]))

    # Deduplicate on the action text.
    unique: list[Recommendation] = []
    seen: set[str] = set()
    for recommendation in recommendations:
        key = " ".join(recommendation.action.lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(recommendation)

    return unique[:8]
