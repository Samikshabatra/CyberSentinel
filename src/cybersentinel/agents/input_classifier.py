"""Input classification agent (LangGraph node 1).

Deliberately deterministic. Routing is a control-flow decision, and control flow
should be reproducible and cheap - an LLM call here would add latency and
non-determinism to every run without improving the decision. This follows the
"do not make every component an LLM" principle: the model is reserved for the
cybersecurity reasoning it was fine-tuned for.
"""

from __future__ import annotations

from cybersentinel.cybersecurity.taxonomy import InputType
from cybersentinel.schemas.analysis import InputClassification
from cybersentinel.utils.validation import (
    extract_indicators,
    looks_like_email,
    looks_like_log,
    looks_like_url_only,
    split_events,
)

# Vocabulary that indicates a vulnerability report rather than an observed event.
_VULNERABILITY_MARKERS = (
    "cve-",
    "cvss",
    "vulnerab",
    "unpatched",
    "patch available",
    "advisory",
    "remote code execution",
)

_NETWORK_MARKERS = (
    "port",
    "packet",
    "firewall",
    "syn",
    "tcp",
    "udp",
    "icmp",
    "inbound",
    "outbound",
    "traffic",
    "netflow",
    "scan",
)


def classify_input(text: str) -> InputClassification:
    """Determine the input type and the indicators that justified the decision."""
    events = split_events(text)
    indicators = extract_indicators(text)
    lowered = text.lower()
    reasons: list[str] = []

    # Single-document formats are checked first. An email always contains a blank
    # line between its headers and body, so testing for multiple events before
    # recognising the format would split every email into two "events".
    if looks_like_url_only(text):
        return InputClassification(
            input_type=InputType.URL,
            confidence=0.95,
            reasoning="Input is a single URL with no surrounding text.",
            indicators=indicators["urls"][:3] or ["single URL"],
            event_count=1,
        )

    # Multi-event submissions take priority over the remaining formats:
    # correlation changes the whole route.
    if len(events) > 1:
        return InputClassification(
            input_type=InputType.MULTI_EVENT,
            confidence=0.9,
            reasoning=f"Input separated into {len(events)} distinct events.",
            indicators=[f"{len(events)} events detected"],
            event_count=len(events),
        )

    if looks_like_email(text):
        reasons.append("email header fields present")
        if indicators["urls"]:
            reasons.append(f"{len(indicators['urls'])} URL(s) in the body")
        return InputClassification(
            input_type=InputType.EMAIL,
            confidence=0.9,
            reasoning="; ".join(reasons),
            indicators=reasons,
            event_count=1,
        )

    vulnerability_hits = [marker for marker in _VULNERABILITY_MARKERS if marker in lowered]
    if vulnerability_hits:
        # A CVE reference is decisive; softer markers need to outweigh log shape.
        decisive = "cve-" in lowered or "cvss" in lowered
        if decisive or not looks_like_log(text):
            return InputClassification(
                input_type=InputType.VULNERABILITY,
                confidence=0.85 if decisive else 0.6,
                reasoning=f"Vulnerability vocabulary present: {', '.join(vulnerability_hits[:3])}.",
                indicators=vulnerability_hits[:5],
                event_count=1,
            )

    if looks_like_log(text):
        return InputClassification(
            input_type=InputType.LOG,
            confidence=0.85,
            reasoning="Multiple timestamped or syslog-shaped lines.",
            indicators=["timestamped log lines"],
            event_count=1,
        )

    network_hits = [marker for marker in _NETWORK_MARKERS if marker in lowered]
    if len(network_hits) >= 2 or indicators["ports"]:
        return InputClassification(
            input_type=InputType.NETWORK_EVENT,
            confidence=0.7,
            reasoning=f"Network vocabulary present: {', '.join(network_hits[:4])}.",
            indicators=network_hits[:5] or ["port reference"],
            event_count=1,
        )

    return InputClassification(
        input_type=InputType.ALERT,
        confidence=0.5,
        reasoning="No specific format detected; treated as a free-text security alert.",
        indicators=[],
        event_count=1,
    )
