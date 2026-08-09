"""Threat taxonomy for CyberSentinel.

The label set is closed: every classification produced anywhere in the system
must normalise to one of these values. `UNKNOWN` is a first-class outcome so the
pipeline is never forced to invent an attack class.
"""

from __future__ import annotations

from enum import StrEnum


class AttackType(StrEnum):
    """Supported threat categories (blueprint section 7)."""

    PHISHING = "Phishing"
    BRUTE_FORCE = "Brute Force"
    CREDENTIAL_ATTACK = "Credential Attack"
    SQL_INJECTION = "SQL Injection"
    XSS = "Cross-Site Scripting"
    MALWARE = "Malware"
    DDOS = "DDoS / Network Attack"
    RECONNAISSANCE = "Port Scanning / Reconnaissance"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    DATA_EXFILTRATION = "Data Exfiltration"
    SUSPICIOUS_AUTH = "Suspicious Authentication Activity"
    VULNERABILITY = "Vulnerability / Exploit"
    INSIDER_THREAT = "Insider Threat"
    BENIGN = "Benign"
    UNKNOWN = "Unknown"


class Severity(StrEnum):
    """Severity levels. `UNKNOWN` is used when evidence is insufficient."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class InputType(StrEnum):
    """Input categories used for conditional routing in the LangGraph workflow."""

    EMAIL = "email"
    LOG = "log"
    URL = "url"
    VULNERABILITY = "vulnerability"
    NETWORK_EVENT = "network_event"
    ALERT = "alert"
    MULTI_EVENT = "multi_event"


class ApprovalDecision(StrEnum):
    """Human-in-the-loop outcomes."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"
    NOT_REQUIRED = "NOT_REQUIRED"


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.UNKNOWN: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

# Aliases the LLM (or a dataset) may emit, mapped onto the canonical label set.
# Keys are lowercase and punctuation-normalised by `_normalise_key`.
_ATTACK_ALIASES: dict[str, AttackType] = {
    "phishing": AttackType.PHISHING,
    "spear phishing": AttackType.PHISHING,
    "spearphishing": AttackType.PHISHING,
    "credential phishing": AttackType.PHISHING,
    "business email compromise": AttackType.PHISHING,
    "bec": AttackType.PHISHING,
    "brute force": AttackType.BRUTE_FORCE,
    "bruteforce": AttackType.BRUTE_FORCE,
    "password spraying": AttackType.BRUTE_FORCE,
    "password guessing": AttackType.BRUTE_FORCE,
    "credential attack": AttackType.CREDENTIAL_ATTACK,
    "credential access": AttackType.CREDENTIAL_ATTACK,
    "credential stuffing": AttackType.CREDENTIAL_ATTACK,
    "credential theft": AttackType.CREDENTIAL_ATTACK,
    "sql injection": AttackType.SQL_INJECTION,
    "sqli": AttackType.SQL_INJECTION,
    "sql injection attempt": AttackType.SQL_INJECTION,
    "cross site scripting": AttackType.XSS,
    "cross-site scripting": AttackType.XSS,
    "xss": AttackType.XSS,
    "malware": AttackType.MALWARE,
    "ransomware": AttackType.MALWARE,
    "trojan": AttackType.MALWARE,
    "worm": AttackType.MALWARE,
    "spyware": AttackType.MALWARE,
    "ddos": AttackType.DDOS,
    "ddos / network attack": AttackType.DDOS,
    "dos": AttackType.DDOS,
    "denial of service": AttackType.DDOS,
    "network attack": AttackType.DDOS,
    "port scanning": AttackType.RECONNAISSANCE,
    "port scan": AttackType.RECONNAISSANCE,
    "reconnaissance": AttackType.RECONNAISSANCE,
    "recon": AttackType.RECONNAISSANCE,
    "scanning": AttackType.RECONNAISSANCE,
    "port scanning / reconnaissance": AttackType.RECONNAISSANCE,
    "privilege escalation": AttackType.PRIVILEGE_ESCALATION,
    "privesc": AttackType.PRIVILEGE_ESCALATION,
    "data exfiltration": AttackType.DATA_EXFILTRATION,
    "exfiltration": AttackType.DATA_EXFILTRATION,
    "data theft": AttackType.DATA_EXFILTRATION,
    "suspicious authentication activity": AttackType.SUSPICIOUS_AUTH,
    "suspicious authentication": AttackType.SUSPICIOUS_AUTH,
    "authentication anomaly": AttackType.SUSPICIOUS_AUTH,
    "anomalous login": AttackType.SUSPICIOUS_AUTH,
    "vulnerability": AttackType.VULNERABILITY,
    "vulnerability / exploit": AttackType.VULNERABILITY,
    "exploit": AttackType.VULNERABILITY,
    "cve": AttackType.VULNERABILITY,
    "insider threat": AttackType.INSIDER_THREAT,
    "insider": AttackType.INSIDER_THREAT,
    "anomalous user behavior": AttackType.INSIDER_THREAT,
    "benign": AttackType.BENIGN,
    "normal": AttackType.BENIGN,
    "benign / normal": AttackType.BENIGN,
    "no threat": AttackType.BENIGN,
    "none": AttackType.BENIGN,
    "unknown": AttackType.UNKNOWN,
    "insufficient evidence": AttackType.UNKNOWN,
    "undetermined": AttackType.UNKNOWN,
}

_SEVERITY_ALIASES: dict[str, Severity] = {
    "low": Severity.LOW,
    "informational": Severity.LOW,
    "info": Severity.LOW,
    "minor": Severity.LOW,
    "medium": Severity.MEDIUM,
    "moderate": Severity.MEDIUM,
    "high": Severity.HIGH,
    "major": Severity.HIGH,
    "critical": Severity.CRITICAL,
    "severe": Severity.CRITICAL,
    "unknown": Severity.UNKNOWN,
    "insufficient evidence": Severity.UNKNOWN,
}


def _normalise_key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def normalise_attack_type(value: str | None) -> AttackType:
    """Map a free-form label onto the closed taxonomy, defaulting to UNKNOWN."""
    if not value:
        return AttackType.UNKNOWN
    key = _normalise_key(value)
    if key in _ATTACK_ALIASES:
        return _ATTACK_ALIASES[key]
    for member in AttackType:
        if _normalise_key(member.value) == key:
            return member
    return AttackType.UNKNOWN


def normalise_severity(value: str | None) -> Severity:
    """Map a free-form severity onto the closed severity set."""
    if not value:
        return Severity.UNKNOWN
    key = _normalise_key(value)
    return _SEVERITY_ALIASES.get(key, Severity.UNKNOWN)


def severity_at_least(candidate: Severity, threshold: Severity) -> bool:
    """True when `candidate` is at or above `threshold` in the severity order."""
    return SEVERITY_ORDER[candidate] >= SEVERITY_ORDER[threshold]


#: Categories used to stratify the fine-tuning dataset.
DATASET_LABELS: tuple[AttackType, ...] = tuple(
    label for label in AttackType if label is not AttackType.UNKNOWN
)
