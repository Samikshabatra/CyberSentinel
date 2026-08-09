"""MITRE ATT&CK / CWE / CVE validation utilities.

This module exists to stop the LLM inventing threat-intelligence identifiers.
It holds a small, hand-verified catalogue of ATT&CK techniques and CWE entries
relevant to the supported threat categories, plus format validators for CVE ids.

Rules enforced here:

* A technique id is only accepted if it exists in the catalogue **and**, when a
  grounding context is supplied, appears in that retrieved context.
* A CVE id is only accepted if it is syntactically valid **and** it appeared in
  the retrieved context - the model may never introduce a CVE on its own.

The catalogue is a validation allowlist, not the knowledge base. Descriptive
content is retrieved through RAG (see `cybersentinel.rag`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cybersentinel.cybersecurity.taxonomy import AttackType

TECHNIQUE_ID_PATTERN = re.compile(r"^T\d{4}(?:\.\d{3})?$")
TACTIC_ID_PATTERN = re.compile(r"^TA\d{4}$")
CVE_ID_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,7}$")
CWE_ID_PATTERN = re.compile(r"^CWE-\d{1,5}$")

ATTACK_TECHNIQUE_URL = "https://attack.mitre.org/techniques/{path}/"
ATTACK_TACTIC_URL = "https://attack.mitre.org/tactics/{tactic_id}/"
CWE_URL = "https://cwe.mitre.org/data/definitions/{number}.html"
NVD_CVE_URL = "https://nvd.nist.gov/vuln/detail/{cve_id}"


@dataclass(frozen=True)
class Technique:
    """A MITRE ATT&CK (Enterprise) technique entry."""

    technique_id: str
    name: str
    tactic: str
    tactic_id: str

    @property
    def url(self) -> str:
        return ATTACK_TECHNIQUE_URL.format(path=self.technique_id.replace(".", "/"))

    def to_dict(self) -> dict[str, str]:
        return {
            "technique_id": self.technique_id,
            "name": self.name,
            "tactic": self.tactic,
            "tactic_id": self.tactic_id,
            "url": self.url,
            "source": "MITRE ATT&CK",
        }


@dataclass(frozen=True)
class Weakness:
    """A CWE entry."""

    cwe_id: str
    name: str

    @property
    def url(self) -> str:
        return CWE_URL.format(number=self.cwe_id.split("-")[1])

    def to_dict(self) -> dict[str, str]:
        return {"cwe_id": self.cwe_id, "name": self.name, "url": self.url, "source": "CWE"}


# --- ATT&CK Enterprise tactics -------------------------------------------------
TACTICS: dict[str, str] = {
    "TA0043": "Reconnaissance",
    "TA0042": "Resource Development",
    "TA0001": "Initial Access",
    "TA0002": "Execution",
    "TA0003": "Persistence",
    "TA0004": "Privilege Escalation",
    "TA0005": "Defense Evasion",
    "TA0006": "Credential Access",
    "TA0007": "Discovery",
    "TA0008": "Lateral Movement",
    "TA0009": "Collection",
    "TA0011": "Command and Control",
    "TA0010": "Exfiltration",
    "TA0040": "Impact",
}

#: Ordering used when rendering a correlated attack chain.
#:
#: This is an *intrusion narrative* order, not the ATT&CK matrix column order.
#: The matrix lists Privilege Escalation before Credential Access because it is
#: a reference layout, not a sequence. In an observed intrusion, credential
#: access typically precedes the access it enables, so the chain reads
#: Reconnaissance -> Discovery -> Credential Access -> Initial Access ->
#: Execution -> Persistence -> Privilege Escalation -> ... -> Impact.
TACTIC_ORDER: tuple[str, ...] = (
    "TA0043",  # Reconnaissance
    "TA0042",  # Resource Development
    "TA0007",  # Discovery
    "TA0006",  # Credential Access
    "TA0001",  # Initial Access
    "TA0002",  # Execution
    "TA0003",  # Persistence
    "TA0004",  # Privilege Escalation
    "TA0005",  # Defense Evasion
    "TA0008",  # Lateral Movement
    "TA0009",  # Collection
    "TA0011",  # Command and Control
    "TA0010",  # Exfiltration
    "TA0040",  # Impact
)


def _technique(technique_id: str, name: str, tactic_id: str) -> Technique:
    return Technique(
        technique_id=technique_id,
        name=name,
        tactic=TACTICS[tactic_id],
        tactic_id=tactic_id,
    )


# --- Verified technique catalogue ---------------------------------------------
TECHNIQUES: dict[str, Technique] = {
    technique.technique_id: technique
    for technique in (
        _technique("T1595", "Active Scanning", "TA0043"),
        _technique("T1595.001", "Active Scanning: Scanning IP Blocks", "TA0043"),
        _technique("T1595.002", "Active Scanning: Vulnerability Scanning", "TA0043"),
        _technique("T1046", "Network Service Discovery", "TA0007"),
        _technique("T1087", "Account Discovery", "TA0007"),
        _technique("T1566", "Phishing", "TA0001"),
        _technique("T1566.001", "Phishing: Spearphishing Attachment", "TA0001"),
        _technique("T1566.002", "Phishing: Spearphishing Link", "TA0001"),
        _technique("T1190", "Exploit Public-Facing Application", "TA0001"),
        _technique("T1078", "Valid Accounts", "TA0001"),
        _technique("T1133", "External Remote Services", "TA0001"),
        _technique("T1204", "User Execution", "TA0002"),
        _technique("T1204.001", "User Execution: Malicious Link", "TA0002"),
        _technique("T1204.002", "User Execution: Malicious File", "TA0002"),
        _technique("T1059", "Command and Scripting Interpreter", "TA0002"),
        _technique("T1110", "Brute Force", "TA0006"),
        _technique("T1110.001", "Brute Force: Password Guessing", "TA0006"),
        _technique("T1110.003", "Brute Force: Password Spraying", "TA0006"),
        _technique("T1110.004", "Brute Force: Credential Stuffing", "TA0006"),
        _technique("T1003", "OS Credential Dumping", "TA0006"),
        _technique("T1555", "Credentials from Password Stores", "TA0006"),
        _technique("T1552", "Unsecured Credentials", "TA0006"),
        _technique("T1056", "Input Capture", "TA0006"),
        _technique("T1068", "Exploitation for Privilege Escalation", "TA0004"),
        _technique("T1548", "Abuse Elevation Control Mechanism", "TA0004"),
        _technique("T1134", "Access Token Manipulation", "TA0004"),
        _technique("T1053", "Scheduled Task/Job", "TA0003"),
        _technique("T1136", "Create Account", "TA0003"),
        _technique("T1070", "Indicator Removal", "TA0005"),
        _technique("T1027", "Obfuscated Files or Information", "TA0005"),
        _technique("T1213", "Data from Information Repositories", "TA0009"),
        _technique("T1005", "Data from Local System", "TA0009"),
        _technique("T1041", "Exfiltration Over C2 Channel", "TA0010"),
        _technique("T1048", "Exfiltration Over Alternative Protocol", "TA0010"),
        _technique("T1567", "Exfiltration Over Web Service", "TA0010"),
        _technique("T1030", "Data Transfer Size Limits", "TA0010"),
        _technique("T1498", "Network Denial of Service", "TA0040"),
        _technique("T1499", "Endpoint Denial of Service", "TA0040"),
        _technique("T1486", "Data Encrypted for Impact", "TA0040"),
        _technique("T1485", "Data Destruction", "TA0040"),
    )
}

# --- Verified CWE catalogue ----------------------------------------------------
WEAKNESSES: dict[str, Weakness] = {
    weakness.cwe_id: weakness
    for weakness in (
        Weakness("CWE-89", "Improper Neutralization of Special Elements used in an SQL Command"),
        Weakness("CWE-79", "Improper Neutralization of Input During Web Page Generation"),
        Weakness("CWE-78", "Improper Neutralization of Special Elements used in an OS Command"),
        Weakness("CWE-22", "Improper Limitation of a Pathname to a Restricted Directory"),
        Weakness("CWE-287", "Improper Authentication"),
        Weakness("CWE-307", "Improper Restriction of Excessive Authentication Attempts"),
        Weakness("CWE-521", "Weak Password Requirements"),
        Weakness("CWE-798", "Use of Hard-coded Credentials"),
        Weakness("CWE-269", "Improper Privilege Management"),
        Weakness("CWE-250", "Execution with Unnecessary Privileges"),
        Weakness("CWE-200", "Exposure of Sensitive Information to an Unauthorized Actor"),
        Weakness("CWE-311", "Missing Encryption of Sensitive Data"),
        Weakness("CWE-352", "Cross-Site Request Forgery"),
        Weakness("CWE-434", "Unrestricted Upload of File with Dangerous Type"),
        Weakness("CWE-400", "Uncontrolled Resource Consumption"),
        Weakness("CWE-770", "Allocation of Resources Without Limits or Throttling"),
    )
}

# --- Category hints -----------------------------------------------------------
# Candidate techniques per category. These seed retrieval queries; they are NOT
# asserted as findings unless supported by evidence or retrieved context.
CATEGORY_TECHNIQUE_HINTS: dict[AttackType, tuple[str, ...]] = {
    AttackType.PHISHING: ("T1566", "T1566.001", "T1566.002", "T1204.001"),
    AttackType.BRUTE_FORCE: ("T1110", "T1110.001", "T1110.003"),
    AttackType.CREDENTIAL_ATTACK: ("T1110.004", "T1003", "T1555", "T1552", "T1078"),
    AttackType.SQL_INJECTION: ("T1190",),
    AttackType.XSS: ("T1190", "T1059"),
    AttackType.MALWARE: ("T1204.002", "T1059", "T1486", "T1027"),
    AttackType.DDOS: ("T1498", "T1499"),
    AttackType.RECONNAISSANCE: ("T1595", "T1595.001", "T1595.002", "T1046"),
    AttackType.PRIVILEGE_ESCALATION: ("T1068", "T1548", "T1134"),
    AttackType.DATA_EXFILTRATION: ("T1041", "T1048", "T1567", "T1030", "T1005"),
    AttackType.SUSPICIOUS_AUTH: ("T1078", "T1110"),
    AttackType.VULNERABILITY: ("T1190", "T1068"),
    AttackType.INSIDER_THREAT: ("T1078", "T1213", "T1005"),
    AttackType.BENIGN: (),
    AttackType.UNKNOWN: (),
}

CATEGORY_CWE_HINTS: dict[AttackType, tuple[str, ...]] = {
    AttackType.SQL_INJECTION: ("CWE-89",),
    AttackType.XSS: ("CWE-79",),
    AttackType.BRUTE_FORCE: ("CWE-307", "CWE-521"),
    AttackType.CREDENTIAL_ATTACK: ("CWE-798", "CWE-287"),
    AttackType.PRIVILEGE_ESCALATION: ("CWE-269", "CWE-250"),
    AttackType.DATA_EXFILTRATION: ("CWE-200", "CWE-311"),
    AttackType.DDOS: ("CWE-400", "CWE-770"),
    AttackType.MALWARE: ("CWE-434",),
    AttackType.SUSPICIOUS_AUTH: ("CWE-287",),
}


# --- Validation helpers -------------------------------------------------------
def is_valid_technique_id(technique_id: str) -> bool:
    """True when the id is well-formed *and* present in the verified catalogue."""
    normalised = technique_id.strip()
    return bool(TECHNIQUE_ID_PATTERN.match(normalised)) and normalised in TECHNIQUES


def is_wellformed_cve_id(cve_id: str) -> bool:
    """True when the id matches the CVE format. Says nothing about existence."""
    return bool(CVE_ID_PATTERN.match(cve_id.strip().upper()))


def is_valid_cwe_id(cwe_id: str) -> bool:
    """True when the id is well-formed and present in the verified catalogue."""
    normalised = cwe_id.strip().upper()
    return bool(CWE_ID_PATTERN.match(normalised)) and normalised in WEAKNESSES


def get_technique(technique_id: str) -> Technique | None:
    return TECHNIQUES.get(technique_id.strip())


def get_weakness(cwe_id: str) -> Weakness | None:
    return WEAKNESSES.get(cwe_id.strip().upper())


def extract_identifiers(text: str) -> dict[str, list[str]]:
    """Pull technique / CVE / CWE identifiers out of free text."""
    techniques = sorted(set(re.findall(r"\bT\d{4}(?:\.\d{3})?\b", text)))
    cves = sorted({match.upper() for match in re.findall(r"\bCVE-\d{4}-\d{4,7}\b", text, re.I)})
    cwes = sorted({match.upper() for match in re.findall(r"\bCWE-\d{1,5}\b", text, re.I)})
    return {"techniques": techniques, "cves": cves, "cwes": cwes}


def filter_grounded_techniques(
    candidate_ids: list[str],
    grounding_text: str = "",
) -> tuple[list[Technique], list[str]]:
    """Split candidate technique ids into grounded techniques and rejected ids.

    A candidate is grounded when it is in the verified catalogue. When
    `grounding_text` (the retrieved RAG context) is supplied, the id must also
    appear there - retrieval is the source of truth for threat intelligence.
    """
    grounded: list[Technique] = []
    rejected: list[str] = []

    for raw_id in candidate_ids:
        technique_id = raw_id.strip()
        technique = TECHNIQUES.get(technique_id)
        if technique is None:
            rejected.append(technique_id)
            continue
        if grounding_text and technique_id not in grounding_text:
            rejected.append(technique_id)
            continue
        if technique not in grounded:
            grounded.append(technique)

    return grounded, rejected


def filter_grounded_cves(
    candidate_ids: list[str],
    grounding_text: str,
) -> tuple[list[str], list[str]]:
    """Accept CVE ids only when well-formed AND present in the retrieved context."""
    accepted: list[str] = []
    rejected: list[str] = []
    upper_context = grounding_text.upper()

    for raw_id in candidate_ids:
        cve_id = raw_id.strip().upper()
        if is_wellformed_cve_id(cve_id) and cve_id in upper_context:
            if cve_id not in accepted:
                accepted.append(cve_id)
        else:
            rejected.append(cve_id)

    return accepted, rejected


def order_by_kill_chain(techniques: list[Technique]) -> list[Technique]:
    """Sort techniques into ATT&CK kill-chain order for attack-chain rendering."""
    return sorted(
        techniques,
        key=lambda technique: (
            TACTIC_ORDER.index(technique.tactic_id)
            if technique.tactic_id in TACTIC_ORDER
            else len(TACTIC_ORDER)
        ),
    )


def cve_reference(cve_id: str) -> dict[str, str]:
    """Build a citation for a CVE id (NVD is the authoritative source)."""
    normalised = cve_id.strip().upper()
    return {"cve_id": normalised, "url": NVD_CVE_URL.format(cve_id=normalised), "source": "NVD"}
