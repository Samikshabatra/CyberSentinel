"""LLM backends.

Two backends are provided:

* ``mock``  - deterministic rule-based generator. No GPU, no downloads, no
  network. This is what makes the application runnable (and testable) without
  retraining, and it is also the ``rules`` baseline in the ablation study.
* ``hf``    - Hugging Face Transformers, optionally with 4-bit quantisation and
  a QLoRA adapter merged in at load time.

Both implement the same ``LLMBackend`` interface, so every agent is written
once and the backend is swapped through configuration.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from cybersentinel.utils.config import Settings, get_settings
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when a backend cannot produce output."""


class LLMBackend(ABC):
    """Chat-completion interface shared by every backend."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the assistant reply for a list of chat messages."""

    def generate_batch(
        self,
        batch: list[list[dict[str, str]]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """Default batch implementation: sequential calls."""
        return [self.generate(messages, max_new_tokens, temperature) for messages in batch]

    @property
    def info(self) -> dict[str, Any]:
        return {"backend": self.name}


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #
class MockBackend(LLMBackend):
    """Deterministic keyword-scoring backend.

    It is honest about its limits: when no rule fires with enough weight it
    returns ``Unknown`` rather than guessing, which is exactly the behaviour the
    hallucination evaluation checks for.
    """

    name = "mock"

    # (attack label, severity, [(pattern, weight)])
    _RULES: tuple[tuple[str, str, tuple[tuple[str, float], ...]], ...] = (
        (
            "Brute Force",
            "HIGH",
            (
                (r"failed (?:ssh )?login", 0.45),
                (r"failed password", 0.45),
                (r"failed sign-?in", 0.45),
                (r"authentication fail", 0.4),
                (r"rejected authentication", 0.45),
                (r"unsuccessful (?:login|authentication|sign-?in)", 0.4),
                (r"invalid password", 0.35),
                (r"\b[\d,]{2,}\s+(?:failed|invalid|rejected)", 0.35),
                (r"one failed sign-?in each", 0.5),
                (r"brute.?force", 0.5),
                (r"password spray", 0.45),
                (r"lockout threshold", 0.25),
                (r"authentication attempts", 0.2),
            ),
        ),
        (
            "Phishing",
            "HIGH",
            (
                (r"verify your account", 0.4),
                (r"urgent", 0.15),
                (r"click (?:here|the link)", 0.3),
                (r"suspend(?:ed|ing)? your account", 0.35),
                (r"mailbox will be suspended", 0.4),
                (r"confirm your credentials", 0.45),
                (r"sign in with your (?:network|corporate)? ?account", 0.35),
                (r"reply-to address", 0.35),
                (r"impersonat", 0.35),
                (r"wire transfer", 0.3),
                (r"reset your password", 0.2),
                (r"phish", 0.5),
                (r"hxxp|\[\.\]", 0.3),
                (r"(?im)^\s*subject:", 0.2),
                (r"(?im)^\s*from:", 0.15),
                (r"remote support tool", 0.3),
            ),
        ),
        (
            "SQL Injection",
            "HIGH",
            (
                (r"union\s+select", 0.5),
                (r"or\s+'?1'?\s*=\s*'?1", 0.5),
                (r"'\s*or\s*'", 0.35),
                (r"drop\s+table", 0.45),
                (r"sleep\(\d+\)", 0.4),
                (r"where clause", 0.2),
                (r"sqlmap", 0.4),
                (r"blind injection", 0.4),
                (r"sql injection", 0.5),
                (r"--\s*$", 0.15),
            ),
        ),
        (
            "Cross-Site Scripting",
            "MEDIUM",
            (
                (r"<script", 0.5),
                (r"javascript:", 0.4),
                (r"onerror\s*=", 0.4),
                (r"onload\s*=", 0.3),
                (r"document\.cookie", 0.45),
                (r"<img src=", 0.3),
                (r"unescaped|without encoding", 0.3),
                (r"alert\(", 0.25),
                (r"\bxss\b|cross-?site scripting", 0.5),
            ),
        ),
        (
            "Malware",
            "CRITICAL",
            (
                (r"ransomware", 0.5),
                (r"malware", 0.45),
                (r"trojan", 0.45),
                (r"shadow cop(?:y|ies)", 0.45),
                (r"quarantined", 0.4),
                (r"antivirus", 0.3),
                (r"mass file (?:modification|rewrite|encryption)", 0.45),
                (r"\.locked\b", 0.4),
                (r"startup registry key", 0.35),
                (r"powershell\.exe|encoded command", 0.35),
                (r"spawned .{0,20}(?:powershell|cmd|wscript|bash|script)", 0.4),
                (r"every \d+ seconds", 0.25),
                (r"beacon", 0.3),
                (r"\bc2\b|command and control", 0.35),
                (r"\b[a-f0-9]{64}\b", 0.2),
            ),
        ),
        (
            "DDoS / Network Attack",
            "HIGH",
            (
                (r"\bddos\b", 0.5),
                (r"denial[- ]of[- ]service", 0.5),
                (r"syn flood", 0.5),
                (r"requests per second", 0.4),
                (r"connection table", 0.4),
                (r"amplification|reflection", 0.4),
                (r"any queries", 0.3),
                (r"spoofed source", 0.35),
                (r"traffic spike|latency has risen", 0.3),
                (r"bandwidth saturat", 0.3),
            ),
        ),
        (
            "Port Scanning / Reconnaissance",
            "MEDIUM",
            (
                (r"port scan", 0.5),
                (r"nmap", 0.45),
                (r"scanning|scanned", 0.3),
                (r"connection attempts", 0.35),
                (r"sequential ports|\d[\d,]* (?:sequential )?ports", 0.4),
                (r"reconnaissance", 0.45),
                (r"/\.env|/wp-login|/phpmyadmin|/admin\b", 0.4),
                (r"content discovery", 0.4),
                (r"contacted port \d+", 0.4),
                (r"on [\d,]+ different (?:internal )?(?:systems|hosts|servers)", 0.35),
                (r"returning 404|returned 404", 0.2),
            ),
        ),
        (
            "Privilege Escalation",
            "CRITICAL",
            (
                (r"privilege escalation", 0.5),
                (r"\bsudo\b.*root", 0.3),
                (r"added to (?:the )?(?:administrators?|domain admins)", 0.5),
                (r"escalat(?:e|ed|ion) .*privileg", 0.45),
                (r"root shell", 0.45),
                (r"setuid", 0.4),
                (r"administrator polic(?:y|ies)", 0.4),
                (r"attaching an administrator", 0.4),
                (r"privileged group", 0.35),
            ),
        ),
        (
            "Data Exfiltration",
            "CRITICAL",
            (
                (r"exfiltrat", 0.5),
                (r"large (?:data )?transfer", 0.35),
                (r"\b[\d.,]+\s?(?:gb|mb)\b", 0.25),
                (r"outbound (?:volume|transfer|traffic)", 0.35),
                (r"uploaded", 0.3),
                (r"file-?sharing service", 0.4),
                (r"base64-?like subdomains|encoded .*subdomain", 0.45),
                (r"data (?:theft|leak)", 0.4),
                (r"archived .*records|customer records", 0.35),
                (r"transferred it to", 0.3),
            ),
        ),
        (
            "Credential Attack",
            "HIGH",
            (
                (r"credential (?:stuffing|dump|theft|access)", 0.5),
                (r"mimikatz", 0.5),
                (r"lsass", 0.5),
                (r"password (?:dump|hash)", 0.35),
                (r"breach corpus|breached credential", 0.5),
                (r"plaintext .*password", 0.45),
                (r"hard-?coded credential", 0.4),
                (r"credential store", 0.4),
            ),
        ),
        (
            "Vulnerability / Exploit",
            "HIGH",
            (
                (r"cve-\d{4}-\d{4,7}", 0.45),
                (r"exploit", 0.3),
                (r"unpatched|outdated", 0.35),
                (r"remote code execution|\brce\b", 0.4),
                (r"vulnerab", 0.35),
                (r"advisory", 0.4),
                (r"\bcvss\b", 0.4),
                (r"vendor patch", 0.35),
                (r"dependency audit", 0.35),
            ),
        ),
        (
            "Suspicious Authentication Activity",
            "MEDIUM",
            (
                (r"impossible travel", 0.5),
                (r"(?:login|signed in) from (?:a )?(?:new|unfamiliar)", 0.4),
                (r"unusual (?:login|sign-?in)", 0.4),
                (r"m(?:ulti-)?f(?:actor)?a? ?(?:fatigue|bypass|push)", 0.4),
                (r"push notifications", 0.4),
                (r"interactively from a workstation", 0.4),
                (r"service account .*sign(?:ed)? in", 0.35),
                (r"outside (?:of )?(?:normal )?(?:business|operating) hours", 0.3),
                (r"signed in from the office .* and again from", 0.45),
            ),
        ),
        (
            "Insider Threat",
            "HIGH",
            (
                (r"insider", 0.45),
                (r"resign(?:ed|ation)", 0.45),
                (r"downloaded [\d,]+ documents", 0.4),
                (r"unrelated to (?:their|his|her) role", 0.5),
                (r"application unrelated", 0.45),
                (r"bulk (?:download|export)", 0.35),
                (r"shared drive", 0.25),
            ),
        ),
        (
            "Benign",
            "LOW",
            (
                (r"successful (?:login|authentication|sign-?in)", 0.25),
                (r"scheduled (?:backup|maintenance|job)", 0.5),
                (r"routine", 0.35),
                (r"no (?:anomal|threat|issue)", 0.4),
                (r"normal (?:weekday )?pattern|matching their normal", 0.45),
                (r"planned (?:package )?upgrade|change calendar", 0.5),
                (r"change (?:record|ticket) exists|help-?desk ticket", 0.4),
                (r"self-service portal", 0.4),
                (r"designated internal|approved internal", 0.4),
                (r"business hours", 0.2),
                (r"usual office", 0.35),
            ),
        ),
    )

    _EVIDENCE_PATTERNS: tuple[tuple[str, str], ...] = (
        (r"\b\d+\s+failed\b[^.\n]*", "repeated authentication failures"),
        (r"\bfrom (?:ip )?(?:\d{1,3}\.){3}\d{1,3}\b", "source IP present in event"),
        (r"within \d+ \w+", "activity compressed into a short time window"),
        (r"\b\d{1,5} ports?\b", "multiple ports touched"),
        (r"cve-\d{4}-\d{4,7}", "vulnerability identifier referenced in the event"),
    )

    _TECHNIQUE_HINT: dict[str, tuple[str, ...]] = {
        "Brute Force": ("T1110",),
        "Phishing": ("T1566",),
        "SQL Injection": ("T1190",),
        "Cross-Site Scripting": ("T1190",),
        "Malware": ("T1204",),
        "DDoS / Network Attack": ("T1498",),
        "Port Scanning / Reconnaissance": ("T1595",),
        "Privilege Escalation": ("T1068",),
        "Data Exfiltration": ("T1041",),
        "Credential Attack": ("T1003",),
        "Vulnerability / Exploit": ("T1190",),
        "Suspicious Authentication Activity": ("T1078",),
        "Insider Threat": ("T1078",),
    }

    #: Below this total score the backend refuses to classify.
    DECISION_THRESHOLD = 0.45

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        if "threat-intelligence assistant" in system:
            return self._intel_response(user)
        if "correlation assistant" in system:
            return self._correlation_response(user)
        if "response advisor" in system:
            return self._response_recommendations(user)
        if "report writer" in system:
            return self._report_response(user)
        return self._detection_response(user)

    # --- task-specific mock responses ---
    def _detection_response(self, user_text: str) -> str:
        import json

        text = user_text.lower()
        scores: dict[str, float] = {}
        severities: dict[str, str] = {}

        for label, severity, patterns in self._RULES:
            score = sum(weight for pattern, weight in patterns if re.search(pattern, text, re.I))
            if score > 0:
                scores[label] = round(score, 3)
                severities[label] = severity

        if not scores or max(scores.values()) < self.DECISION_THRESHOLD:
            return json.dumps(
                {
                    "attack_type": "Unknown",
                    "severity": "UNKNOWN",
                    "confidence": round(max(scores.values(), default=0.0), 2),
                    "evidence": [],
                    "candidate_techniques": [],
                    "reasoning": (
                        "No rule matched with sufficient weight. Insufficient evidence to "
                        "assert an attack category."
                    ),
                }
            )

        label = max(scores, key=lambda key: scores[key])
        confidence = min(0.95, round(0.45 + scores[label] / 2, 2))

        evidence = [
            description
            for pattern, description in self._EVIDENCE_PATTERNS
            if re.search(pattern, text, re.I)
        ]
        if not evidence:
            evidence = [f"Event text matches known '{label}' indicators."]

        return json.dumps(
            {
                "attack_type": label,
                "severity": severities[label],
                "confidence": confidence,
                "evidence": evidence,
                "candidate_techniques": list(self._TECHNIQUE_HINT.get(label, ())),
                "reasoning": (
                    f"Rule-based backend matched '{label}' indicators with a combined weight of "
                    f"{scores[label]}. This is a heuristic baseline, not a fine-tuned judgement."
                ),
            }
        )

    def _intel_response(self, user_text: str) -> str:
        """Echo back only identifiers that literally appear in the retrieved context."""
        import json

        context_marker = "Retrieved context:"
        context = user_text.split(context_marker, 1)[1] if context_marker in user_text else ""
        techniques = sorted(set(re.findall(r"\bT\d{4}(?:\.\d{3})?\b", context)))
        cwes = sorted({match.upper() for match in re.findall(r"\bCWE-\d{1,5}\b", context, re.I)})
        cves = sorted({match.upper() for match in re.findall(r"\bCVE-\d{4}-\d{4,7}\b", context, re.I)})
        return json.dumps(
            {
                "techniques": techniques[:5],
                "cwe": cwes[:3],
                "cve": cves[:3],
                "justification": (
                    "Identifiers copied from the retrieved context only."
                    if techniques or cwes or cves
                    else "Retrieved context does not support any mapping."
                ),
                "grounded": bool(techniques or cwes or cves),
            }
        )

    def _correlation_response(self, user_text: str) -> str:
        import json

        # The deterministic correlation engine does the real work; the mock LLM
        # only confirms whether shared indicators were reported.
        shared = "none" not in user_text.split("Shared indicators across events:")[-1].lower()
        return json.dumps(
            {
                "is_correlated": shared,
                "confidence": 0.6 if shared else 0.1,
                "attack_chain": [],
                "summary": (
                    "Events share at least one indicator, which is consistent with a single "
                    "actor but not proof of one."
                    if shared
                    else "No shared indicators were found across the submitted events."
                ),
            }
        )

    def _response_recommendations(self, user_text: str) -> str:
        import json

        attack_line = re.search(r"attack_type:\s*(.+)", user_text)
        attack = attack_line.group(1).strip() if attack_line else "Unknown"
        playbook = _FALLBACK_PLAYBOOK.get(attack, _FALLBACK_PLAYBOOK["default"])
        return json.dumps({"recommendations": playbook})

    def _report_response(self, user_text: str) -> str:
        import json

        attack = re.search(r"attack_type:\s*(.+)", user_text)
        severity = re.search(r"severity:\s*(.+)", user_text)
        label = attack.group(1).strip() if attack else "Unknown"
        level = severity.group(1).strip() if severity else "UNKNOWN"
        if label == "Unknown":
            summary = (
                "The submitted event did not contain enough evidence to assert a threat "
                "category. No attack classification is claimed. An analyst should review the "
                "raw event and supply additional context or correlated events."
            )
            detected = "No conclusive threat was identified."
        else:
            summary = (
                f"The event is consistent with a probable {label} at {level} severity. "
                "The finding is derived from indicators present in the submitted event and, "
                "where available, from retrieved threat-intelligence references. "
                "All recommendations require analyst validation before action."
            )
            detected = f"Activity consistent with {label}."
        return json.dumps(
            {
                "summary": summary,
                "what_was_detected": detected,
                "why": "Indicators in the event matched the reported category.",
                "next_steps": "Validate the evidence and action the prioritised recommendations.",
            }
        )


#: Deterministic defensive playbooks. Used by the mock backend and as the
#: fallback whenever the response agent fails - the system must always be able
#: to give an analyst something safe to do.
_FALLBACK_PLAYBOOK: dict[str, list[dict[str, Any]]] = {
    "Brute Force": [
        {
            "action": "Review authentication logs for the source IP to confirm whether any attempt succeeded",
            "rationale": "Distinguishes a failed campaign from an actual compromise.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Enable or tighten authentication rate limiting and account lockout thresholds",
            "rationale": "Directly reduces the feasibility of repeated guessing.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Enforce multi-factor authentication on the targeted accounts",
            "rationale": "Password guessing alone no longer yields access.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Block the source IP at the perimeter firewall",
            "rationale": "Stops the ongoing attempts from this origin.",
            "priority": "HIGH",
            "high_impact": True,
        },
    ],
    "Phishing": [
        {
            "action": "Preserve the original message with full headers for forensic analysis",
            "rationale": "Headers establish the true sender path and support takedown requests.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Search the mail gateway for other recipients of the same campaign",
            "rationale": "Phishing is rarely delivered to a single mailbox.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Check whether any recipient submitted credentials to the linked page",
            "rationale": "Determines whether this is an attempted or successful credential theft.",
            "priority": "CRITICAL",
            "high_impact": False,
        },
        {
            "action": "Reset credentials for any user who interacted with the link",
            "rationale": "Invalidates credentials that may already be in attacker hands.",
            "priority": "CRITICAL",
            "high_impact": True,
        },
    ],
    "SQL Injection": [
        {
            "action": "Review application and database logs for the affected endpoint and parameter",
            "rationale": "Establishes whether the payload reached the database and what it returned.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Replace dynamic SQL on the affected endpoint with parameterised queries",
            "rationale": "Removes the underlying weakness rather than filtering symptoms.",
            "priority": "CRITICAL",
            "high_impact": False,
        },
        {
            "action": "Verify the database account used by the application has least-privilege rights",
            "rationale": "Limits the blast radius of any successful injection.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Add a WAF rule for the observed payload pattern as a temporary control",
            "rationale": "Buys time while the code fix is deployed.",
            "priority": "MEDIUM",
            "high_impact": False,
        },
    ],
    "Cross-Site Scripting": [
        {
            "action": "Identify the vulnerable parameter and apply context-aware output encoding",
            "rationale": "Encoding at the point of rendering is the reliable fix.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Deploy or tighten a Content-Security-Policy header on the affected application",
            "rationale": "Reduces the impact of any injected script that slips through.",
            "priority": "MEDIUM",
            "high_impact": False,
        },
        {
            "action": "Review session handling and set HttpOnly and Secure flags on session cookies",
            "rationale": "Limits session theft, the usual goal of a stored XSS payload.",
            "priority": "MEDIUM",
            "high_impact": False,
        },
    ],
    "Malware": [
        {
            "action": "Preserve volatile evidence and take a forensic image of the affected host",
            "rationale": "Containment destroys evidence; capture it first.",
            "priority": "CRITICAL",
            "high_impact": False,
        },
        {
            "action": "Search the estate for the observed file hashes and network indicators",
            "rationale": "Determines the true scope before remediation begins.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Isolate the affected host from the network",
            "rationale": "Prevents lateral movement and further command-and-control traffic.",
            "priority": "CRITICAL",
            "high_impact": True,
        },
    ],
    "Data Exfiltration": [
        {
            "action": "Identify the destination and volume of the outbound transfer from network flow data",
            "rationale": "Establishes what left the environment and where it went.",
            "priority": "CRITICAL",
            "high_impact": False,
        },
        {
            "action": "Review the acting account's recent access to sensitive repositories",
            "rationale": "Scopes which data classes are implicated.",
            "priority": "CRITICAL",
            "high_impact": False,
        },
        {
            "action": "Engage legal and data-protection stakeholders on notification obligations",
            "rationale": "Regulatory clocks may start at the point of discovery.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Revoke the acting account's active sessions and tokens",
            "rationale": "Stops an in-progress transfer.",
            "priority": "CRITICAL",
            "high_impact": True,
        },
    ],
    "Privilege Escalation": [
        {
            "action": "Review the audit trail for the privilege change and identify who authorised it",
            "rationale": "Separates a legitimate administrative change from an attack.",
            "priority": "CRITICAL",
            "high_impact": False,
        },
        {
            "action": "Inventory actions taken by the account after the privilege change",
            "rationale": "Escalation is a means; the impact lies in what followed.",
            "priority": "CRITICAL",
            "high_impact": False,
        },
        {
            "action": "Revert the unauthorised group membership or role assignment",
            "rationale": "Restores the intended authorisation model.",
            "priority": "CRITICAL",
            "high_impact": True,
        },
    ],
    "Port Scanning / Reconnaissance": [
        {
            "action": "Confirm whether the scanned services are intentionally internet-exposed",
            "rationale": "Reconnaissance matters most where real attack surface exists.",
            "priority": "MEDIUM",
            "high_impact": False,
        },
        {
            "action": "Check whether the same source later attempted authentication or exploitation",
            "rationale": "Scanning followed by access attempts indicates a progressing intrusion.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Verify perimeter rules restrict management services to trusted ranges",
            "rationale": "Reduces the surface the next scan will find.",
            "priority": "MEDIUM",
            "high_impact": False,
        },
    ],
    "DDoS / Network Attack": [
        {
            "action": "Characterise the traffic by protocol, source distribution and request pattern",
            "rationale": "Mitigation choice depends on the attack shape.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Engage the upstream provider or DDoS mitigation service",
            "rationale": "Volumetric attacks must be absorbed before the edge.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Apply rate limiting at the edge for the abusive traffic pattern",
            "rationale": "Preserves capacity for legitimate users.",
            "priority": "HIGH",
            "high_impact": True,
        },
    ],
    "Credential Attack": [
        {
            "action": "Identify which credentials were accessed and where they are valid",
            "rationale": "Scopes the reset that follows.",
            "priority": "CRITICAL",
            "high_impact": False,
        },
        {
            "action": "Rotate the affected credentials and any service accounts sharing them",
            "rationale": "Stolen credentials remain usable until rotated.",
            "priority": "CRITICAL",
            "high_impact": True,
        },
        {
            "action": "Review authentication logs for successful use of the affected credentials",
            "rationale": "Detects whether the theft has already been acted upon.",
            "priority": "HIGH",
            "high_impact": False,
        },
    ],
    "Vulnerability / Exploit": [
        {
            "action": "Confirm whether the referenced component and version are present in the estate",
            "rationale": "Applicability determines urgency.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Apply the vendor patch or documented mitigation to affected systems",
            "rationale": "Removes the exploitable condition.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Review logs for exploitation attempts against the affected component",
            "rationale": "Patching does not undo a pre-existing compromise.",
            "priority": "HIGH",
            "high_impact": False,
        },
    ],
    "Suspicious Authentication Activity": [
        {
            "action": "Contact the account owner to confirm whether the sign-in was expected",
            "rationale": "The cheapest and most reliable way to resolve the ambiguity.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Review the session for actions taken after sign-in",
            "rationale": "Distinguishes benign travel from account takeover.",
            "priority": "HIGH",
            "high_impact": False,
        },
        {
            "action": "Require re-authentication with MFA for the account",
            "rationale": "Cheap containment while the review completes.",
            "priority": "MEDIUM",
            "high_impact": False,
        },
    ],
    "Insider Threat": [
        {
            "action": "Preserve access logs and file-transfer records before notifying anyone",
            "rationale": "Evidence integrity matters most in personnel cases.",
            "priority": "CRITICAL",
            "high_impact": False,
        },
        {
            "action": "Engage HR and legal before taking any action affecting the individual",
            "rationale": "Insider cases carry employment-law obligations.",
            "priority": "CRITICAL",
            "high_impact": False,
        },
        {
            "action": "Review whether the accessed data is consistent with the person's role",
            "rationale": "Establishes whether the access was actually anomalous.",
            "priority": "HIGH",
            "high_impact": False,
        },
    ],
    "Benign": [
        {
            "action": "No response action required; retain the event for baseline tuning",
            "rationale": "Confirmed benign events improve future detection thresholds.",
            "priority": "LOW",
            "high_impact": False,
        }
    ],
    "default": [
        {
            "action": "Collect additional context for the event, including surrounding log entries",
            "rationale": "The current evidence is insufficient for a confident classification.",
            "priority": "MEDIUM",
            "high_impact": False,
        },
        {
            "action": "Check whether the observed indicators appear in previous incidents",
            "rationale": "Historical context often resolves an ambiguous single event.",
            "priority": "MEDIUM",
            "high_impact": False,
        },
        {
            "action": "Escalate to a senior analyst if the activity recurs",
            "rationale": "Repetition changes an isolated anomaly into a pattern.",
            "priority": "MEDIUM",
            "high_impact": False,
        },
    ],
}


def fallback_playbook(attack_type: str) -> list[dict[str, Any]]:
    """Return the deterministic defensive playbook for an attack category."""
    return [dict(item) for item in _FALLBACK_PLAYBOOK.get(attack_type, _FALLBACK_PLAYBOOK["default"])]


# --------------------------------------------------------------------------- #
# Hugging Face backend
# --------------------------------------------------------------------------- #
class HuggingFaceBackend(LLMBackend):
    """Transformers backend with optional 4-bit quantisation and LoRA adapter.

    Heavy imports happen inside ``_load`` so that importing this module (and
    therefore running the test suite) never requires torch.
    """

    name = "hf"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded_adapter: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LLMError(
                "The 'hf' backend requires the ML extra. Install with: pip install -e '.[ml]'"
            ) from exc

        settings = self.settings
        logger.info(f"loading base model {settings.base_model_name}")

        quantization_config = None
        if settings.llm_load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except (ImportError, RuntimeError) as exc:  # pragma: no cover
                logger.warning(f"4-bit quantisation unavailable, loading in fp16/bf16: {exc}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            settings.base_model_name, token=settings.hf_token
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.base_model_name,
            quantization_config=quantization_config,
            dtype=dtype,
            device_map=settings.llm_device if torch.cuda.is_available() else None,
            token=settings.hf_token,
        )

        adapter_path = settings.model_adapter_path
        if adapter_path:
            resolved = settings.resolve(adapter_path)
            if not resolved.exists():
                logger.warning(f"adapter path does not exist, running base model: {resolved}")
            else:
                from peft import PeftModel

                logger.info(f"attaching LoRA adapter from {resolved}")
                self._model = PeftModel.from_pretrained(self._model, str(resolved))
                self._loaded_adapter = str(resolved)

        self._model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self._load()
        import torch

        settings = self.settings
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        resolved_temperature = settings.llm_temperature if temperature is None else temperature
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or settings.llm_max_new_tokens,
                temperature=max(resolved_temperature, 1e-4),
                do_sample=resolved_temperature > 0,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    @property
    def info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "base_model": self.settings.base_model_name,
            "adapter": self._loaded_adapter,
            "loaded": self._model is not None,
        }


def build_backend(settings: Settings | None = None, backend: str | None = None) -> LLMBackend:
    """Construct a backend, falling back to mock when the ML stack is missing."""
    resolved_settings = settings or get_settings()
    name = backend or resolved_settings.llm_backend

    if name == "mock":
        return MockBackend()
    if name == "hf":
        return HuggingFaceBackend(resolved_settings)
    raise LLMError(f"unknown LLM backend: {name}")


@lru_cache(maxsize=4)
def get_backend(backend: str | None = None) -> LLMBackend:
    """Return a cached backend instance (models are expensive to load)."""
    return build_backend(backend=backend)
